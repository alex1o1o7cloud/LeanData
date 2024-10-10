import Mathlib

namespace intersection_point_l3087_308769

/-- The quadratic function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The point (0, -1) -/
def point : ℝ × ℝ := (0, -1)

/-- Theorem: The point (0, -1) is the intersection point of y = x^2 - 1 with the y-axis -/
theorem intersection_point :
  (point.1 = 0) ∧ 
  (point.2 = f point.1) ∧ 
  (∀ x : ℝ, x ≠ point.1 → (x, f x) ≠ point) :=
by sorry

end intersection_point_l3087_308769


namespace student_community_selection_l3087_308717

/-- The number of ways to select communities for students. -/
def ways_to_select (num_students : ℕ) (num_communities : ℕ) : ℕ :=
  num_communities ^ num_students

/-- Theorem: Given 4 students and 3 communities, where each student chooses 1 community,
    the number of different ways of selection is 3^4. -/
theorem student_community_selection :
  ways_to_select 4 3 = 3^4 := by
  sorry

#eval ways_to_select 4 3  -- Should output 81

end student_community_selection_l3087_308717


namespace min_value_and_integral_bound_l3087_308733

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * x * Real.log x

-- Define the integral G
def G (a b : ℝ) : ℝ := ∫ x in a..b, |Real.log x - Real.log ((a + b) / 2)|

-- State the theorem
theorem min_value_and_integral_bound 
  (k : ℝ) (a b : ℝ) (h1 : k ≠ 0) (h2 : 0 < a) (h3 : a < b) :
  (∃ (x : ℝ), f k x = -1 / Real.exp 1 ∧ ∀ (y : ℝ), f k y ≥ -1 / Real.exp 1) →
  (k = 1 ∧ 
   G a b = a * Real.log a + b * Real.log b - (a + b) * Real.log ((a + b) / 2) ∧
   G a b / (b - a) < Real.log 2) := by
  sorry

end

end min_value_and_integral_bound_l3087_308733


namespace bank_balance_deduction_l3087_308736

theorem bank_balance_deduction (X : ℝ) (current_balance : ℝ) : 
  current_balance = X * 0.9 ∧ current_balance = 90000 → X = 100000 := by
sorry

end bank_balance_deduction_l3087_308736


namespace molecular_weight_7_moles_AlOH3_l3087_308713

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of aluminum atoms in Al(OH)3 -/
def num_Al : ℕ := 1

/-- The number of oxygen atoms in Al(OH)3 -/
def num_O : ℕ := 3

/-- The number of hydrogen atoms in Al(OH)3 -/
def num_H : ℕ := 3

/-- The number of moles of Al(OH)3 -/
def num_moles : ℝ := 7

/-- The molecular weight of Al(OH)3 in g/mol -/
def molecular_weight_AlOH3 : ℝ :=
  num_Al * atomic_weight_Al + num_O * atomic_weight_O + num_H * atomic_weight_H

theorem molecular_weight_7_moles_AlOH3 :
  num_moles * molecular_weight_AlOH3 = 546.07 := by sorry

end molecular_weight_7_moles_AlOH3_l3087_308713


namespace complex_symmetry_product_l3087_308716

theorem complex_symmetry_product : 
  ∀ (z₁ z₂ : ℂ), 
  z₁ = 3 + 2*I → 
  (z₂.re = z₁.im ∧ z₂.im = z₁.re) → 
  z₁ * z₂ = 13*I := by
sorry

end complex_symmetry_product_l3087_308716


namespace triangle_cos_C_l3087_308783

/-- Given a triangle ABC where b = 2a and b sin A = c sin C, prove that cos C = 3/4 -/
theorem triangle_cos_C (a b c : ℝ) (A B C : ℝ) : 
  b = 2 * a → b * Real.sin A = c * Real.sin C → Real.cos C = 3 / 4 := by
  sorry

end triangle_cos_C_l3087_308783


namespace largest_x_value_l3087_308785

theorem largest_x_value : ∃ x : ℝ,
  (15 * x^2 - 30 * x + 9) / (4 * x - 3) + 6 * x = 7 * x - 2 ∧
  x = (19 + Real.sqrt 229) / 22 ∧
  ∀ y : ℝ, (15 * y^2 - 30 * y + 9) / (4 * y - 3) + 6 * y = 7 * y - 2 → y ≤ x :=
by sorry

end largest_x_value_l3087_308785


namespace jose_join_time_l3087_308738

/-- Represents the problem of determining when Jose joined Tom's business --/
theorem jose_join_time (tom_investment jose_investment total_profit jose_profit : ℚ) 
  (h1 : tom_investment = 3000)
  (h2 : jose_investment = 4500)
  (h3 : total_profit = 5400)
  (h4 : jose_profit = 3000) :
  let x := (12 * tom_investment * (total_profit - jose_profit)) / 
           (jose_investment * jose_profit) - 12
  x = 2 := by sorry

end jose_join_time_l3087_308738


namespace potatoes_already_cooked_l3087_308777

/-- Given a chef cooking potatoes with the following conditions:
  - The total number of potatoes to cook is 13
  - Each potato takes 6 minutes to cook
  - It will take 48 minutes to cook the remaining potatoes
  This theorem proves that the number of potatoes already cooked is 5. -/
theorem potatoes_already_cooked 
  (total_potatoes : ℕ) 
  (cooking_time_per_potato : ℕ) 
  (remaining_cooking_time : ℕ) 
  (h1 : total_potatoes = 13)
  (h2 : cooking_time_per_potato = 6)
  (h3 : remaining_cooking_time = 48) :
  total_potatoes - (remaining_cooking_time / cooking_time_per_potato) = 5 :=
by sorry

end potatoes_already_cooked_l3087_308777


namespace katyas_classmates_l3087_308768

theorem katyas_classmates :
  ∀ (N : ℕ) (K : ℕ),
    (K + 10 - (N + 1)) / (N + 1) = K + 1 →
    N > 0 →
    N = 9 := by
  sorry

end katyas_classmates_l3087_308768


namespace sequence_2024th_term_l3087_308774

/-- Definition of the sequence term -/
def sequenceTerm (n : ℕ) : ℤ × ℕ := ((-1)^(n+1) * (2*n - 1), n)

/-- The 2024th term of the sequence -/
def term2024 : ℤ × ℕ := sequenceTerm 2024

/-- Theorem stating the 2024th term of the sequence -/
theorem sequence_2024th_term :
  term2024 = (-4047, 2024) := by sorry

end sequence_2024th_term_l3087_308774


namespace sin_value_from_tan_cos_l3087_308784

theorem sin_value_from_tan_cos (θ : Real) 
  (h1 : 6 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : π < θ) (h3 : θ < 2 * π) : 
  Real.sin θ = 1/2 := by
  sorry

end sin_value_from_tan_cos_l3087_308784


namespace third_cube_edge_l3087_308712

-- Define the cube volume function
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

-- Define the given cubes
def cube1_edge : ℝ := 4
def cube2_edge : ℝ := 5
def final_cube_edge : ℝ := 6

-- Theorem statement
theorem third_cube_edge :
  ∃ (third_edge : ℝ),
    cube_volume third_edge + cube_volume cube1_edge + cube_volume cube2_edge
    = cube_volume final_cube_edge ∧ third_edge = 3 := by
  sorry

end third_cube_edge_l3087_308712


namespace function_periodicity_l3087_308743

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (10 - x) = 4

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_periodicity (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_cond : satisfies_condition f) : 
  periodic f 20 := by sorry

end function_periodicity_l3087_308743


namespace muffin_banana_cost_ratio_l3087_308706

/-- The cost ratio of a muffin to a banana given Susie and Calvin's purchases -/
theorem muffin_banana_cost_ratio :
  ∀ (m b c : ℚ),
  (5 * m + 2 * b + 3 * c = 1) →  -- Normalize Susie's purchase to 1
  (4 * m + 18 * b + c = 3) →     -- Calvin's purchase is 3 times Susie's
  (c = 2 * b) →                  -- A cookie costs twice as much as a banana
  (m / b = 4 / 11) :=
by sorry

end muffin_banana_cost_ratio_l3087_308706


namespace train_speed_l3087_308790

/-- Given a train of length 125 meters crossing a bridge of length 250 meters in 30 seconds,
    its speed is 45 km/hr. -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 125 →
  bridge_length = 250 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_l3087_308790


namespace pet_ownership_percentage_l3087_308793

theorem pet_ownership_percentage (total_students : ℕ) (both_pets : ℕ)
  (h1 : total_students = 500)
  (h2 : both_pets = 50) :
  (both_pets : ℚ) / total_students * 100 = 10 := by
  sorry

end pet_ownership_percentage_l3087_308793


namespace x_range_restriction_l3087_308767

-- Define a monotonically decreasing function on (0, +∞)
def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

-- Define the main theorem
theorem x_range_restriction 
  (f : ℝ → ℝ) 
  (h_monotonic : monotonically_decreasing f)
  (h_condition : ∀ x, 0 < x → f x < f (2*x - 2)) :
  ∀ x, (0 < x ∧ f x < f (2*x - 2)) → 1 < x ∧ x < 2 :=
sorry

end x_range_restriction_l3087_308767


namespace message_reconstruction_existence_l3087_308739

/-- Represents a text as a list of characters -/
def Text := List Char

/-- Represents a permutation of characters -/
def Permutation := Char → Char

/-- Represents a substitution of characters -/
def Substitution := Char → Char

/-- Apply a permutation to a text -/
def applyPermutation (p : Permutation) (t : Text) : Text :=
  t.map p

/-- Apply a substitution to a text -/
def applySubstitution (s : Substitution) (t : Text) : Text :=
  t.map s

/-- Check if a substitution is bijective -/
def isBijectiveSubstitution (s : Substitution) : Prop :=
  Function.Injective s ∧ Function.Surjective s

theorem message_reconstruction_existence :
  ∃ (original : Text) (p : Permutation) (s : Substitution),
    let text1 := "МИМОПРАСТЕТИРАСИСПДАИСАФЕИИБОЕТКЖРГЛЕОЛОИШИСАННСЙСАООЛТЛЕЯТУИЦВЫИПИЯДПИЩПЬПСЕЮЯ".data
    let text2 := "УЩФМШПДРЕЦЧЕШЮЧДАКЕЧМДВКШБЕЕЧДФЭПЙЩГШФЩЦЕЮЩФПМЕЧПМРРМЕОЧХЕШРГИФРЯЯЛКДФФЕЕ".data
    applyPermutation p original = text1 ∧
    applySubstitution s original = text2 ∧
    isBijectiveSubstitution s ∧
    original = "ШЕСТАЯОЛИМПИАДАПОКРИПТОГРАФИИПОСВЯЩЕННАЯСЕМЬДЕСЯТИПЯТИЛЕТИЮСПЕЦИАЛЬНОЙСЛУЖБЫРОССИИ".data :=
by sorry


end message_reconstruction_existence_l3087_308739


namespace linear_function_second_quadrant_increasing_l3087_308740

/-- A linear function passing through the second quadrant with increasing y as x increases -/
def LinearFunctionSecondQuadrantIncreasing (k b : ℝ) : Prop :=
  k > 0 ∧ b > 0

/-- The property of a function passing through the second quadrant -/
def PassesThroughSecondQuadrant (f : ℝ → ℝ) : Prop :=
  ∃ x y, x < 0 ∧ y > 0 ∧ f x = y

/-- The property of a function being increasing -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- Theorem stating that a linear function with positive slope and y-intercept
    passes through the second quadrant and is increasing -/
theorem linear_function_second_quadrant_increasing (k b : ℝ) :
  LinearFunctionSecondQuadrantIncreasing k b ↔
  PassesThroughSecondQuadrant (λ x => k * x + b) ∧
  IsIncreasing (λ x => k * x + b) :=
by sorry

end linear_function_second_quadrant_increasing_l3087_308740


namespace min_semi_focal_distance_l3087_308714

/-- The minimum semi-focal distance of a hyperbola satisfying certain conditions -/
theorem min_semi_focal_distance (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  c^2 = a^2 + b^2 →  -- Definition of semi-focal distance for hyperbola
  (1/3 * c + 1) * c = a * b →  -- Condition on distance from origin to line
  c ≥ 6 := by sorry

end min_semi_focal_distance_l3087_308714


namespace set_membership_implies_value_l3087_308705

theorem set_membership_implies_value (a : ℝ) : 
  3 ∈ ({a, a^2 - 2*a} : Set ℝ) → a = -1 := by
  sorry

end set_membership_implies_value_l3087_308705


namespace existence_of_single_root_quadratic_l3087_308721

/-- Given a quadratic polynomial with leading coefficient 1 and exactly one root,
    there exists a point (p, q) such that x^2 + px + q also has exactly one root. -/
theorem existence_of_single_root_quadratic 
  (b c : ℝ) 
  (h1 : b^2 - 4*c = 0) : 
  ∃ p q : ℝ, p^2 - 4*q = 0 := by
sorry

end existence_of_single_root_quadratic_l3087_308721


namespace complex_number_location_l3087_308737

theorem complex_number_location (z : ℂ) (h : z * Complex.I = 2 + 3 * Complex.I) :
  0 < z.re ∧ z.im < 0 := by sorry

end complex_number_location_l3087_308737


namespace Q_on_circle_25_line_AB_equation_l3087_308731

-- Define the circle P
def circle_P (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define that Q is outside circle P
def Q_outside_P (a b : ℝ) : Prop := a^2 + b^2 > 16

-- Define circle M with diameter PQ intersecting circle P at A and B
def circle_M_intersects_P (a b : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), circle_P x1 y1 ∧ circle_P x2 y2 ∧ 
  ((x1 - a)^2 + (y1 - b)^2 = (x1^2 + y1^2) / 4) ∧
  ((x2 - a)^2 + (y2 - b)^2 = (x2^2 + y2^2) / 4)

-- Theorem 1: When QA = QB = 3, Q lies on x^2 + y^2 = 25
theorem Q_on_circle_25 (a b : ℝ) 
  (h1 : Q_outside_P a b) 
  (h2 : circle_M_intersects_P a b) 
  (h3 : ∃ (x1 y1 x2 y2 : ℝ), circle_P x1 y1 ∧ circle_P x2 y2 ∧
        (x1 - a)^2 + (y1 - b)^2 = 9 ∧ (x2 - a)^2 + (y2 - b)^2 = 9) :
  a^2 + b^2 = 25 :=
sorry

-- Theorem 2: When Q(4, 6), the equation of line AB is 2x + 3y - 8 = 0
theorem line_AB_equation 
  (h1 : Q_outside_P 4 6) 
  (h2 : circle_M_intersects_P 4 6) :
  ∃ (x1 y1 x2 y2 : ℝ), circle_P x1 y1 ∧ circle_P x2 y2 ∧
  2 * x1 + 3 * y1 - 8 = 0 ∧ 2 * x2 + 3 * y2 - 8 = 0 :=
sorry

end Q_on_circle_25_line_AB_equation_l3087_308731


namespace charm_cost_calculation_l3087_308722

/-- The cost of a single charm used in Tim's necklace business -/
def charm_cost : ℚ := 15

/-- The number of charms used in each necklace -/
def charms_per_necklace : ℕ := 10

/-- The selling price of each necklace -/
def necklace_price : ℚ := 200

/-- The number of necklaces sold -/
def necklaces_sold : ℕ := 30

/-- The total profit from selling 30 necklaces -/
def total_profit : ℚ := 1500

theorem charm_cost_calculation :
  charm_cost * (charms_per_necklace : ℚ) * (necklaces_sold : ℚ) =
  necklace_price * (necklaces_sold : ℚ) - total_profit := by sorry

end charm_cost_calculation_l3087_308722


namespace calculate_expression_l3087_308701

theorem calculate_expression : (-1)^2 + (1/3)^0 = 2 := by sorry

end calculate_expression_l3087_308701


namespace optimal_purchase_plan_l3087_308707

/-- Represents a machine model with its cost and production capacity -/
structure MachineModel where
  cost : ℕ
  production : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  modelA : ℕ
  modelB : ℕ

def modelA : MachineModel := ⟨60000, 15⟩
def modelB : MachineModel := ⟨40000, 10⟩

def totalMachines : ℕ := 10
def budgetLimit : ℕ := 440000
def requiredProduction : ℕ := 102

def isValidPlan (plan : PurchasePlan) : Prop :=
  plan.modelA + plan.modelB = totalMachines ∧
  plan.modelA * modelA.cost + plan.modelB * modelB.cost ≤ budgetLimit ∧
  plan.modelA * modelA.production + plan.modelB * modelB.production ≥ requiredProduction

def isOptimalPlan (plan : PurchasePlan) : Prop :=
  isValidPlan plan ∧
  ∀ (otherPlan : PurchasePlan), 
    isValidPlan otherPlan → 
    plan.modelA * modelA.cost + plan.modelB * modelB.cost ≤ 
    otherPlan.modelA * modelA.cost + otherPlan.modelB * modelB.cost

theorem optimal_purchase_plan :
  ∃ (plan : PurchasePlan), isOptimalPlan plan ∧ plan.modelA = 1 ∧ plan.modelB = 9 := by
  sorry

end optimal_purchase_plan_l3087_308707


namespace circle_tangent_to_parabola_directrix_l3087_308794

theorem circle_tangent_to_parabola_directrix (p : ℝ) : 
  p > 0 → 
  (∃ x y : ℝ, x^2 + y^2 - 6*x - 7 = 0 ∧ 
              y^2 = 2*p*x ∧ 
              x = -p) → 
  p = 2 := by
sorry

end circle_tangent_to_parabola_directrix_l3087_308794


namespace polygon_sides_count_l3087_308766

/-- Theorem: For a convex polygon where the sum of the interior angles is twice the sum of its exterior angles, the number of sides is 6. -/
theorem polygon_sides_count (n : ℕ) : n > 2 → (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end polygon_sides_count_l3087_308766


namespace repeating_decimal_equals_fraction_l3087_308724

/-- Represents the repeating decimal 0.42̄157 -/
def repeating_decimal : ℚ := 42157 / 100000 + (157 / 100000) / (1 - 1/1000)

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 4207359 / 99900

/-- Theorem stating that the repeating decimal 0.42̄157 is equal to 4207359/99900 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end repeating_decimal_equals_fraction_l3087_308724


namespace journey_distance_l3087_308730

/-- Represents the man's journey with different speeds and times for each segment -/
structure Journey where
  flat_walk_speed : ℝ
  downhill_run_speed : ℝ
  hilly_walk_speed : ℝ
  hilly_run_speed : ℝ
  flat_walk_time : ℝ
  downhill_run_time : ℝ
  hilly_walk_time : ℝ
  hilly_run_time : ℝ

/-- Calculates the total distance traveled during the journey -/
def total_distance (j : Journey) : ℝ :=
  j.flat_walk_speed * j.flat_walk_time +
  j.downhill_run_speed * j.downhill_run_time +
  j.hilly_walk_speed * j.hilly_walk_time +
  j.hilly_run_speed * j.hilly_run_time

/-- Theorem stating that the total distance traveled is 90 km -/
theorem journey_distance :
  let j : Journey := {
    flat_walk_speed := 8,
    downhill_run_speed := 24,
    hilly_walk_speed := 6,
    hilly_run_speed := 18,
    flat_walk_time := 3,
    downhill_run_time := 1.5,
    hilly_walk_time := 2,
    hilly_run_time := 1
  }
  total_distance j = 90 := by sorry

end journey_distance_l3087_308730


namespace y_value_at_50_l3087_308708

/-- A line passing through given points -/
structure Line where
  -- Define the line using two points
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Theorem: Y-coordinate when X is 50 on a specific line -/
theorem y_value_at_50 (l : Line) (u : ℝ) : 
  l.x1 = 10 ∧ l.y1 = 30 ∧ 
  l.x2 = 15 ∧ l.y2 = 45 ∧ 
  (∃ y3 : ℝ, y3 = 3 * 20 ∧ Line.mk 10 30 20 y3 = l) ∧
  (∃ y4 : ℝ, y4 = u ∧ Line.mk 10 30 40 y4 = l) →
  (∃ y : ℝ, y = 150 ∧ Line.mk 10 30 50 y = l) :=
by sorry

end y_value_at_50_l3087_308708


namespace tempo_original_value_l3087_308760

/-- The original value of a tempo given its insured value and insurance extent --/
theorem tempo_original_value 
  (insured_value : ℝ) 
  (insurance_extent : ℝ) 
  (h1 : insured_value = 70000) 
  (h2 : insurance_extent = 4/5) : 
  ∃ (original_value : ℝ), 
    original_value = 87500 ∧ 
    insured_value = insurance_extent * original_value :=
by
  sorry

#check tempo_original_value

end tempo_original_value_l3087_308760


namespace sum_of_common_roots_l3087_308755

theorem sum_of_common_roots (a : ℝ) :
  (∃ x : ℝ, x^2 + (2*a - 5)*x + a^2 + 1 = 0 ∧ 
            x^3 + (2*a - 5)*x^2 + (a^2 + 1)*x + a^2 - 4 = 0) →
  (∃ x y : ℝ, x^2 - 9*x + 5 = 0 ∧ y^2 - 9*y + 5 = 0 ∧ x + y = 9) :=
by sorry

end sum_of_common_roots_l3087_308755


namespace tamika_drove_farther_l3087_308772

-- Define the given conditions
def tamika_time : ℝ := 8
def tamika_speed : ℝ := 45
def logan_time : ℝ := 5
def logan_speed : ℝ := 55

-- Define the distance calculation function
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

-- Theorem statement
theorem tamika_drove_farther : 
  distance tamika_time tamika_speed - distance logan_time logan_speed = 85 := by
  sorry

end tamika_drove_farther_l3087_308772


namespace ana_number_puzzle_l3087_308715

theorem ana_number_puzzle (x : ℝ) : (((x + 3) * 3 - 4) / 2) = 10 → x = 5 := by
  sorry

end ana_number_puzzle_l3087_308715


namespace six_students_like_no_option_l3087_308758

/-- Represents the food preferences in a class --/
structure FoodPreferences where
  total_students : ℕ
  french_fries : ℕ
  burgers : ℕ
  pizza : ℕ
  tacos : ℕ
  fries_burgers : ℕ
  fries_pizza : ℕ
  fries_tacos : ℕ
  burgers_pizza : ℕ
  burgers_tacos : ℕ
  pizza_tacos : ℕ
  fries_burgers_pizza : ℕ
  fries_burgers_tacos : ℕ
  fries_pizza_tacos : ℕ
  burgers_pizza_tacos : ℕ
  all_four : ℕ

/-- Calculates the number of students who don't like any food option --/
def studentsLikingNoOption (prefs : FoodPreferences) : ℕ :=
  prefs.total_students -
  (prefs.french_fries + prefs.burgers + prefs.pizza + prefs.tacos -
   prefs.fries_burgers - prefs.fries_pizza - prefs.fries_tacos -
   prefs.burgers_pizza - prefs.burgers_tacos - prefs.pizza_tacos +
   prefs.fries_burgers_pizza + prefs.fries_burgers_tacos +
   prefs.fries_pizza_tacos + prefs.burgers_pizza_tacos -
   prefs.all_four)

/-- Theorem: Given the food preferences, 6 students don't like any option --/
theorem six_students_like_no_option (prefs : FoodPreferences)
  (h1 : prefs.total_students = 35)
  (h2 : prefs.french_fries = 20)
  (h3 : prefs.burgers = 15)
  (h4 : prefs.pizza = 18)
  (h5 : prefs.tacos = 12)
  (h6 : prefs.fries_burgers = 10)
  (h7 : prefs.fries_pizza = 8)
  (h8 : prefs.fries_tacos = 6)
  (h9 : prefs.burgers_pizza = 7)
  (h10 : prefs.burgers_tacos = 5)
  (h11 : prefs.pizza_tacos = 9)
  (h12 : prefs.fries_burgers_pizza = 4)
  (h13 : prefs.fries_burgers_tacos = 3)
  (h14 : prefs.fries_pizza_tacos = 2)
  (h15 : prefs.burgers_pizza_tacos = 1)
  (h16 : prefs.all_four = 1) :
  studentsLikingNoOption prefs = 6 := by
  sorry


end six_students_like_no_option_l3087_308758


namespace gcd_problem_l3087_308797

theorem gcd_problem (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (x y : ℕ+), Nat.gcd x y = 10 ∧ Nat.gcd (12 * x) (18 * y) = 60) ∧
  (∀ (c d : ℕ+), Nat.gcd c d = 10 → Nat.gcd (12 * c) (18 * d) ≥ 60) :=
sorry

end gcd_problem_l3087_308797


namespace total_salary_proof_l3087_308723

def salary_B : ℝ := 232

def salary_A : ℝ := 1.5 * salary_B

def total_salary : ℝ := salary_A + salary_B

theorem total_salary_proof : total_salary = 580 := by
  sorry

end total_salary_proof_l3087_308723


namespace quadratic_function_theorem_l3087_308780

/-- A quadratic function passing through three given points -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_theorem :
  ∃ (a b c : ℝ),
    (QuadraticFunction a b c (-2) = 9) ∧
    (QuadraticFunction a b c 0 = 3) ∧
    (QuadraticFunction a b c 4 = 3) ∧
    (∀ x, QuadraticFunction a b c x = (1/2) * x^2 - 2 * x + 3) ∧
    (let vertex_x := -b / (2*a);
     let vertex_y := QuadraticFunction a b c vertex_x;
     vertex_x = 2 ∧ vertex_y = 1) ∧
    (∀ m : ℝ,
      let y₁ := QuadraticFunction a b c m;
      let y₂ := QuadraticFunction a b c (m+1);
      (m < 3/2 → y₁ > y₂) ∧
      (m = 3/2 → y₁ = y₂) ∧
      (m > 3/2 → y₁ < y₂)) := by
  sorry


end quadratic_function_theorem_l3087_308780


namespace square_side_length_l3087_308756

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 6)
  (h2 : rectangle_length = 18) : ∃ square_side : ℝ,
  square_side = 12 ∧ 4 * square_side = 2 * (rectangle_width + rectangle_length) := by
  sorry

end square_side_length_l3087_308756


namespace poly_sequence_properties_l3087_308744

/-- Represents a polynomial sequence generated by the given operation -/
def PolySequence (a : ℝ) (n : ℕ) : List ℝ :=
  sorry

/-- The product of all polynomials in the sequence after n operations -/
def PolyProduct (a : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- The sum of all polynomials in the sequence after n operations -/
def PolySum (a : ℝ) (n : ℕ) : ℝ :=
  sorry

theorem poly_sequence_properties (a : ℝ) :
  (∀ a, |a| ≥ 2 → PolyProduct a 2 ≤ 0) ∧
  (∀ n, PolySum a n = 2*a + 2*(n+1)) :=
by sorry

end poly_sequence_properties_l3087_308744


namespace no_m_exists_for_equality_subset_condition_l3087_308764

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Statement 1: There does not exist an m such that P = S(m)
theorem no_m_exists_for_equality : ¬∃ m : ℝ, P = S m := by sorry

-- Statement 2: For all m ≥ 3, P ⊆ S(m)
theorem subset_condition (m : ℝ) (h : m ≥ 3) : P ⊆ S m := by sorry

end no_m_exists_for_equality_subset_condition_l3087_308764


namespace complement_intersection_theorem_l3087_308770

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_theorem :
  (A ∩ B)ᶜ = {1, 2} :=
by sorry

end complement_intersection_theorem_l3087_308770


namespace special_function_properties_l3087_308798

/-- A function satisfying f(x+y) = f(x) + f(y) for all x, y, and f(x) > 0 for x > 0 -/
class SpecialFunction (f : ℝ → ℝ) :=
  (add : ∀ x y : ℝ, f (x + y) = f x + f y)
  (pos : ∀ x : ℝ, x > 0 → f x > 0)

/-- The main theorem stating that a SpecialFunction is odd and monotonically increasing -/
theorem special_function_properties (f : ℝ → ℝ) [SpecialFunction f] :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end special_function_properties_l3087_308798


namespace first_graders_count_l3087_308778

/-- The number of Kindergarteners to be checked -/
def kindergarteners : ℕ := 26

/-- The number of second graders to be checked -/
def second_graders : ℕ := 20

/-- The number of third graders to be checked -/
def third_graders : ℕ := 25

/-- The time in minutes it takes to check one student -/
def check_time : ℕ := 2

/-- The total time in hours available for all checks -/
def total_time_hours : ℕ := 3

/-- Calculate the number of first graders that need to be checked -/
def first_graders_to_check : ℕ :=
  (total_time_hours * 60 - (kindergarteners + second_graders + third_graders) * check_time) / check_time

/-- Theorem stating that the number of first graders to be checked is 19 -/
theorem first_graders_count : first_graders_to_check = 19 := by
  sorry

end first_graders_count_l3087_308778


namespace f_of_5_equals_105_l3087_308757

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + x

-- State the theorem
theorem f_of_5_equals_105 : f 5 = 105 := by
  sorry

end f_of_5_equals_105_l3087_308757


namespace smallest_population_with_conditions_l3087_308771

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem smallest_population_with_conditions : 
  ∃ n : ℕ, 
    is_perfect_square n ∧ 
    (∃ k : ℕ, n + 150 = k^2 + 1) ∧ 
    is_perfect_square (n + 300) ∧
    n = 144 ∧
    ∀ m : ℕ, m < n → 
      ¬(is_perfect_square m ∧ 
        (∃ k : ℕ, m + 150 = k^2 + 1) ∧ 
        is_perfect_square (m + 300)) :=
by sorry

end smallest_population_with_conditions_l3087_308771


namespace expansion_equality_l3087_308792

theorem expansion_equality (y : ℝ) : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end expansion_equality_l3087_308792


namespace sum_of_odd_powers_l3087_308720

theorem sum_of_odd_powers (x y z a : ℝ) (k : ℕ) 
  (h1 : x + y + z = a) 
  (h2 : x^3 + y^3 + z^3 = a^3) 
  (h3 : Odd k) : 
  x^k + y^k + z^k = a^k := by
  sorry

end sum_of_odd_powers_l3087_308720


namespace nellie_legos_l3087_308700

theorem nellie_legos (L : ℕ) : 
  L - 57 - 24 = 299 → L = 380 := by
sorry

end nellie_legos_l3087_308700


namespace complex_equation_solution_l3087_308726

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I : ℂ) * 2 = (1 : ℂ) + (Complex.I : ℂ) * 2 * a + b → a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l3087_308726


namespace students_in_both_clubs_l3087_308709

theorem students_in_both_clubs 
  (total_students : ℕ) 
  (drama_students : ℕ) 
  (science_students : ℕ) 
  (drama_or_science_students : ℕ) 
  (h1 : total_students = 500)
  (h2 : drama_students = 150)
  (h3 : science_students = 200)
  (h4 : drama_or_science_students = 300) :
  drama_students + science_students - drama_or_science_students = 50 := by
sorry

end students_in_both_clubs_l3087_308709


namespace b_current_age_l3087_308749

/-- Given two people A and B, where in 10 years A will be twice as old as B was 10 years ago,
    and A is currently 7 years older than B, prove that B's current age is 37 years. -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) → 
  (a = b + 7) → 
  b = 37 := by
sorry

end b_current_age_l3087_308749


namespace group_average_age_problem_l3087_308787

theorem group_average_age_problem (n : ℕ) : 
  (n * 14 + 32 = 16 * (n + 1)) → n = 8 := by
  sorry

end group_average_age_problem_l3087_308787


namespace odd_number_between_bounds_l3087_308754

theorem odd_number_between_bounds (N : ℕ) : 
  N % 2 = 1 → (9.5 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 10.5) → N = 39 ∨ N = 41 := by
  sorry

end odd_number_between_bounds_l3087_308754


namespace circle_radii_sum_l3087_308751

theorem circle_radii_sum (r R : ℝ) : 
  r > 0 → R > 0 →  -- Radii are positive
  R - r = 5 →  -- Distance between centers
  π * R^2 - π * r^2 = 100 * π →  -- Area between circles
  r + R = 20 := by
sorry

end circle_radii_sum_l3087_308751


namespace star_three_five_l3087_308765

-- Define the star operation
def star (c d : ℝ) : ℝ := c^2 - 2*c*d + d^2

-- Theorem statement
theorem star_three_five : star 3 5 = 4 := by
  sorry

end star_three_five_l3087_308765


namespace root_product_theorem_l3087_308729

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 + x₁^2 + 1 = 0) → 
  (x₂^5 + x₂^2 + 1 = 0) → 
  (x₃^5 + x₃^2 + 1 = 0) → 
  (x₄^5 + x₄^2 + 1 = 0) → 
  (x₅^5 + x₅^2 + 1 = 0) → 
  (x₁^3 - 2) * (x₂^3 - 2) * (x₃^3 - 2) * (x₄^3 - 2) * (x₅^3 - 2) = -243 := by
sorry

end root_product_theorem_l3087_308729


namespace min_sum_with_constraints_min_sum_achieved_l3087_308761

theorem min_sum_with_constraints (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) (h_sum_sq : x^2 + y^2 + z^2 ≥ 90) : 
  x + y + z ≥ 16 := by
  sorry

theorem min_sum_achieved (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) (h_sum_sq : x^2 + y^2 + z^2 ≥ 90) : 
  ∃ (a b c : ℝ), a ≥ 4 ∧ b ≥ 5 ∧ c ≥ 6 ∧ a^2 + b^2 + c^2 ≥ 90 ∧ a + b + c = 16 := by
  sorry

end min_sum_with_constraints_min_sum_achieved_l3087_308761


namespace curve_properties_l3087_308752

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

def curveC (a : ℝ) : ParametricCurve :=
  { x := λ t => 1 + 2*t,
    y := λ t => a*t^2 }

def pointOnCurve (p : Point2D) (c : ParametricCurve) : Prop :=
  ∃ t : ℝ, c.x t = p.x ∧ c.y t = p.y

theorem curve_properties (a : ℝ) :
  pointOnCurve ⟨3, 1⟩ (curveC a) →
  (a = 1) ∧
  (∀ x y : ℝ, (x - 1)^2 = 4*y ↔ pointOnCurve ⟨x, y⟩ (curveC a)) :=
by sorry

end curve_properties_l3087_308752


namespace max_draws_for_cmwmc_l3087_308747

/-- Represents the number of tiles of each letter in the bag -/
structure TileCounts :=
  (c : Nat)
  (m : Nat)
  (w : Nat)

/-- Represents the number of tiles needed to spell the word -/
structure WordCounts :=
  (c : Nat)
  (m : Nat)
  (w : Nat)

/-- The maximum number of tiles that need to be drawn -/
def maxDraws (bag : TileCounts) (word : WordCounts) : Nat :=
  bag.c + bag.m + bag.w - (word.c - 1) - (word.m - 1) - (word.w - 1)

/-- Theorem stating the maximum number of draws for the given problem -/
theorem max_draws_for_cmwmc :
  let bag := TileCounts.mk 8 8 8
  let word := WordCounts.mk 2 2 1
  maxDraws bag word = 18 := by
  sorry

end max_draws_for_cmwmc_l3087_308747


namespace symmetric_function_domain_l3087_308788

/-- A function with either odd or even symmetry -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∨ (∀ x, f x = -f (-x))

/-- The theorem stating that if a symmetric function is defined on [3-a, 5], then a = -2 -/
theorem symmetric_function_domain (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc (3 - a) 5, f x ≠ 0 → True) →
  SymmetricFunction f →
  a = -2 := by
  sorry

end symmetric_function_domain_l3087_308788


namespace complex_equation_solution_l3087_308728

theorem complex_equation_solution (c d x : ℂ) : 
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 6 * Complex.I →
  x = 3 * Real.sqrt 21 :=
sorry

end complex_equation_solution_l3087_308728


namespace reflection_of_M_l3087_308746

/-- Reflection of a point about the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem reflection_of_M :
  let M : ℝ × ℝ := (3, 2)
  reflect_x M = (3, -2) := by sorry

end reflection_of_M_l3087_308746


namespace sally_final_count_l3087_308742

def sally_pokemon_cards (initial : ℕ) (from_dan : ℕ) (bought : ℕ) : ℕ :=
  initial + from_dan + bought

theorem sally_final_count :
  sally_pokemon_cards 27 41 20 = 88 := by
  sorry

end sally_final_count_l3087_308742


namespace f_monotone_increasing_l3087_308796

open Real

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 - Real.sqrt 3 * cos x * cos (x + π / 2)

theorem f_monotone_increasing :
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 3 → f x < f y :=
by sorry

end f_monotone_increasing_l3087_308796


namespace fraction_simplification_l3087_308750

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (x^2 + x) / (x^2 - 1) = x / (x - 1) := by
  sorry

end fraction_simplification_l3087_308750


namespace stock_market_value_l3087_308763

/-- Calculates the market value of a stock given its income, interest rate, and brokerage fee. -/
def market_value (income : ℚ) (interest_rate : ℚ) (brokerage_rate : ℚ) : ℚ :=
  let face_value := (income * 100) / interest_rate
  let brokerage_fee := (face_value / 100) * brokerage_rate
  face_value - brokerage_fee

/-- Theorem stating that the market value of the stock is 7182 given the specified conditions. -/
theorem stock_market_value :
  market_value 756 10.5 0.25 = 7182 :=
by sorry

end stock_market_value_l3087_308763


namespace vector_equation_solution_l3087_308775

theorem vector_equation_solution (e₁ e₂ : ℝ × ℝ) (x y : ℝ) 
  (h_not_collinear : ¬ ∃ (k : ℝ), e₂ = k • e₁)
  (h_equation : (3*x - 4*y) • e₁ + (2*x - 3*y) • e₂ = 6 • e₁ + 3 • e₂) :
  x - y = 3 := by
sorry

end vector_equation_solution_l3087_308775


namespace num_cases_hearts_D_l3087_308799

/-- The number of cards in a standard deck without jokers -/
def totalCards : ℕ := 52

/-- The number of people among whom the cards are distributed -/
def numPeople : ℕ := 4

/-- The total number of hearts in the deck -/
def totalHearts : ℕ := 13

/-- The number of hearts A has -/
def heartsA : ℕ := 5

/-- The number of hearts B has -/
def heartsB : ℕ := 4

/-- Theorem stating the number of possible cases for D's hearts -/
theorem num_cases_hearts_D : 
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ k : ℕ, k ≤ totalHearts - heartsA - heartsB → 
    (∃ (heartsC heartsD : ℕ), 
      heartsC + heartsD = totalHearts - heartsA - heartsB ∧
      heartsD = k)) ∧
  (∀ k : ℕ, k > totalHearts - heartsA - heartsB → 
    ¬∃ (heartsC heartsD : ℕ), 
      heartsC + heartsD = totalHearts - heartsA - heartsB ∧
      heartsD = k) :=
by sorry

end num_cases_hearts_D_l3087_308799


namespace skating_rink_visitors_l3087_308711

/-- The number of people at a skating rink at noon, given the initial number of visitors,
    the number of people who left, and the number of new arrivals. -/
def people_at_noon (initial : ℕ) (left : ℕ) (arrived : ℕ) : ℕ :=
  initial - left + arrived

/-- Theorem stating that the number of people at the skating rink at noon is 280,
    given the specific values from the problem. -/
theorem skating_rink_visitors : people_at_noon 264 134 150 = 280 := by
  sorry

end skating_rink_visitors_l3087_308711


namespace negation_of_universal_statement_l3087_308710

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) := by
  sorry

end negation_of_universal_statement_l3087_308710


namespace arithmetic_sqrt_of_nine_l3087_308702

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- Theorem statement
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end arithmetic_sqrt_of_nine_l3087_308702


namespace smaller_field_area_l3087_308732

/-- Given a field of 500 hectares divided into two parts, where the difference
    of the areas is one-fifth of their average, the area of the smaller part
    is 225 hectares. -/
theorem smaller_field_area (x y : ℝ) (h1 : x + y = 500)
    (h2 : y - x = (1 / 5) * ((x + y) / 2)) : x = 225 := by
  sorry

end smaller_field_area_l3087_308732


namespace ellipse_center_locus_l3087_308727

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Represents a right angle -/
structure RightAngle where
  vertex : Point

/-- Predicate to check if an ellipse touches both sides of a right angle -/
def touches_right_angle (e : Ellipse) (ra : RightAngle) : Prop :=
  sorry

/-- The locus of the center of the ellipse -/
def center_locus (ra : RightAngle) (a b : ℝ) : Set Point :=
  {p : Point | ∃ e : Ellipse, e.center = p ∧ e.semi_major_axis = a ∧ e.semi_minor_axis = b ∧ touches_right_angle e ra}

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is on an arc of a circle -/
def on_circle_arc (p : Point) (c : Circle) : Prop :=
  sorry

theorem ellipse_center_locus (ra : RightAngle) (a b : ℝ) :
  ∃ c : Circle, c.center = ra.vertex ∧ ∀ p ∈ center_locus ra a b, on_circle_arc p c :=
sorry

end ellipse_center_locus_l3087_308727


namespace josh_shopping_cost_l3087_308735

def film_cost : ℕ := 5
def book_cost : ℕ := 4
def cd_cost : ℕ := 3

def num_films : ℕ := 9
def num_books : ℕ := 4
def num_cds : ℕ := 6

theorem josh_shopping_cost : 
  (num_films * film_cost + num_books * book_cost + num_cds * cd_cost) = 79 := by
  sorry

end josh_shopping_cost_l3087_308735


namespace vector_problem_l3087_308782

/-- Custom vector operation ⊗ -/
def vector_op (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

/-- Theorem statement -/
theorem vector_problem (p q : ℝ × ℝ) 
  (h1 : p = (1, 2)) 
  (h2 : vector_op p q = (-3, -4)) : 
  q = (-3, -2) := by
  sorry

end vector_problem_l3087_308782


namespace quadratic_form_coefficients_l3087_308789

theorem quadratic_form_coefficients :
  let f : ℝ → ℝ := λ x => 2 * x * (x - 1) - 3 * x
  ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ a = 2 ∧ b = -5 ∧ c = 0 :=
by sorry

end quadratic_form_coefficients_l3087_308789


namespace intersection_of_M_and_S_l3087_308703

-- Define the set M
def M : Set ℕ := {x | 0 < x ∧ x < 4}

-- Define the set S
def S : Set ℕ := {2, 3, 5}

-- Theorem statement
theorem intersection_of_M_and_S : M ∩ S = {2, 3} := by
  sorry

end intersection_of_M_and_S_l3087_308703


namespace vector_OC_on_angle_bisector_l3087_308791

/-- Given points A and B, and a point C on the angle bisector of ∠AOB with |OC| = 2,
    prove that OC is equal to the specified vector. -/
theorem vector_OC_on_angle_bisector (A B C : ℝ × ℝ) : 
  A = (0, 1) →
  B = (-3, 4) →
  C.1^2 + C.2^2 = 4 →  -- |OC| = 2
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    C = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧
    t * (A.1^2 + A.2^2) = (1 - t) * (B.1^2 + B.2^2)) →  -- C is on the angle bisector
  C = (-Real.sqrt 10 / 5, 3 * Real.sqrt 10 / 5) :=
by sorry

end vector_OC_on_angle_bisector_l3087_308791


namespace product_abcd_l3087_308781

theorem product_abcd (a b c d : ℚ) : 
  (2*a + 4*b + 6*c + 8*d = 48) →
  (4*(d+c) = b) →
  (4*b + 2*c = a) →
  (c + 1 = d) →
  (a * b * c * d = -319603200 / 10503489) := by
sorry

end product_abcd_l3087_308781


namespace complex_equation_system_l3087_308725

theorem complex_equation_system (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (eq1 : p = (q + r) / (s - 3))
  (eq2 : q = (p + r) / (t - 3))
  (eq3 : r = (p + q) / (u - 3))
  (eq4 : s * t + s * u + t * u = 10)
  (eq5 : s + t + u = 6) :
  s * t * u = 11 := by
  sorry

end complex_equation_system_l3087_308725


namespace special_ellipse_equation_l3087_308704

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The length of the major axis is twice the length of the minor axis -/
  major_twice_minor : ℝ → ℝ → Prop
  /-- The ellipse passes through the point (2, -6) -/
  passes_through_2_neg6 : ℝ → ℝ → Prop
  /-- The ellipse passes through the point (3, 0) -/
  passes_through_3_0 : ℝ → ℝ → Prop
  /-- The eccentricity of the ellipse is √6/3 -/
  eccentricity_sqrt6_div_3 : ℝ → ℝ → Prop

/-- The standard equation of an ellipse -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating that a SpecialEllipse satisfies one of two standard equations -/
theorem special_ellipse_equation (e : SpecialEllipse) :
  (∀ x y, standard_equation 3 (Real.sqrt 3) x y) ∨
  (∀ x y, standard_equation (Real.sqrt 27) 3 x y) :=
sorry

end special_ellipse_equation_l3087_308704


namespace inequality_proof_l3087_308748

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a|

-- State the theorem
theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_solution_set : Set.Icc 0 2 = {x | f x ((1/m) + (1/(2*n))) ≤ 1}) : 
  m + 4*n ≥ 2*Real.sqrt 2 + 3 := by
sorry

end inequality_proof_l3087_308748


namespace martin_improvement_l3087_308795

/-- Represents Martin's cycling performance --/
structure CyclingPerformance where
  laps : ℕ
  time : ℕ

/-- Calculates the time per lap given a cycling performance --/
def timePerLap (performance : CyclingPerformance) : ℚ :=
  performance.time / performance.laps

/-- Martin's initial cycling performance --/
def initialPerformance : CyclingPerformance :=
  { laps := 15, time := 45 }

/-- Martin's improved cycling performance --/
def improvedPerformance : CyclingPerformance :=
  { laps := 18, time := 42 }

/-- Theorem stating the improvement in Martin's per-lap time --/
theorem martin_improvement :
  timePerLap initialPerformance - timePerLap improvedPerformance = 2/3 := by
  sorry

#eval timePerLap initialPerformance - timePerLap improvedPerformance

end martin_improvement_l3087_308795


namespace sufficient_not_necessary_l3087_308745

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 2 → x + y ≥ 3) ∧
  (∃ x y : ℝ, x + y ≥ 3 ∧ ¬(x ≥ 1 ∧ y ≥ 2)) :=
by sorry

end sufficient_not_necessary_l3087_308745


namespace binary_to_quaternary_conversion_l3087_308776

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : List (Fin 4) := sorry

/-- The binary representation of 1110010110₂ -/
def binary_num : List Bool := [true, true, true, false, false, true, false, true, true, false]

/-- The expected quaternary representation of 32112₄ -/
def expected_quaternary : List (Fin 4) := [3, 2, 1, 1, 2]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_num) = expected_quaternary := by sorry

end binary_to_quaternary_conversion_l3087_308776


namespace x_sixth_minus_six_x_squared_l3087_308762

theorem x_sixth_minus_six_x_squared (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end x_sixth_minus_six_x_squared_l3087_308762


namespace equation_represents_hyperbola_l3087_308773

/-- The equation 3x^2 - 9y^2 - 18y = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0),
    ∀ (x y : ℝ), 3 * x^2 - 9 * y^2 - 18 * y = 0 ↔
      ((y + c)^2 / a^2) - (x^2 / b^2) = 1 :=
by sorry

end equation_represents_hyperbola_l3087_308773


namespace quadratic_discriminant_l3087_308719

/-- The discriminant of a quadratic equation ax² + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 2x² + (3 - 1/2)x + 1/2 -/
def a : ℚ := 2
def b : ℚ := 3 - 1/2
def c : ℚ := 1/2

theorem quadratic_discriminant : discriminant a b c = 9/4 := by sorry

end quadratic_discriminant_l3087_308719


namespace complex_number_in_third_quadrant_l3087_308734

theorem complex_number_in_third_quadrant :
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_third_quadrant_l3087_308734


namespace orthocenter_of_triangle_l3087_308779

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (3/2, 5/2, 6). -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (4, 4, 2)
  let C : ℝ × ℝ × ℝ := (3, 5, 6)
  orthocenter A B C = (3/2, 5/2, 6) := by
  sorry

end orthocenter_of_triangle_l3087_308779


namespace apple_difference_apple_problem_solution_l3087_308718

/-- Proves that Mark has 13 fewer apples than Susan given the conditions of the problem -/
theorem apple_difference : ℕ → Prop := fun total_apples =>
  ∀ (greg_sarah_apples susan_apples mark_apples mom_pie_apples mom_leftover_apples : ℕ),
    greg_sarah_apples = 18 →
    susan_apples = 2 * (greg_sarah_apples / 2) →
    mom_pie_apples = 40 →
    mom_leftover_apples = 9 →
    total_apples = mom_pie_apples + mom_leftover_apples →
    mark_apples = total_apples - susan_apples →
    susan_apples - mark_apples = 13

/-- The main theorem stating that there exists a total number of apples satisfying the conditions -/
theorem apple_problem_solution : ∃ total_apples : ℕ, apple_difference total_apples := by
  sorry

end apple_difference_apple_problem_solution_l3087_308718


namespace max_value_after_operations_l3087_308759

def initial_numbers : List ℕ := [1, 2, 3]
def num_operations : ℕ := 9

def operation (numbers : List ℕ) : List ℕ :=
  let sum := numbers.sum
  let max := numbers.maximum?
  match max with
  | none => numbers
  | some m => (sum - m) :: (numbers.filter (· ≠ m))

def iterate_operation (n : ℕ) (numbers : List ℕ) : List ℕ :=
  match n with
  | 0 => numbers
  | n + 1 => iterate_operation n (operation numbers)

theorem max_value_after_operations :
  (iterate_operation num_operations initial_numbers).maximum? = some 233 :=
sorry

end max_value_after_operations_l3087_308759


namespace simplify_expression_l3087_308741

theorem simplify_expression (x : ℝ) : (3*x)^4 + (4*x)*(x^5) = 81*x^4 + 4*x^6 := by
  sorry

end simplify_expression_l3087_308741


namespace expression_equality_l3087_308786

theorem expression_equality (v u w : ℝ) 
  (h1 : u = 3 * v) 
  (h2 : w = 5 * u) : 
  2 * v + u + w = 20 * v := by sorry

end expression_equality_l3087_308786


namespace expression_upper_bound_l3087_308753

theorem expression_upper_bound :
  ∃ (U : ℕ), 
    (∃ (S : Finset ℤ), 
      (Finset.card S = 50) ∧ 
      (∀ n ∈ S, 1 < 4*n + 7 ∧ 4*n + 7 < U) ∧
      (∀ U' < U, ∃ n ∈ S, 4*n + 7 ≥ U')) →
    U = 204 :=
by sorry

end expression_upper_bound_l3087_308753
