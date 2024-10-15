import Mathlib

namespace NUMINAMATH_CALUDE_mike_seashells_count_l3257_325703

/-- The number of seashells Mike found initially -/
def initial_seashells : ℝ := 6.0

/-- The number of seashells Mike found later -/
def later_seashells : ℝ := 4.0

/-- The total number of seashells Mike found -/
def total_seashells : ℝ := initial_seashells + later_seashells

theorem mike_seashells_count : total_seashells = 10.0 := by sorry

end NUMINAMATH_CALUDE_mike_seashells_count_l3257_325703


namespace NUMINAMATH_CALUDE_average_age_combined_l3257_325705

-- Define the groups and their properties
def num_fifth_graders : ℕ := 40
def avg_age_fifth_graders : ℚ := 12
def num_parents : ℕ := 60
def avg_age_parents : ℚ := 35
def num_teachers : ℕ := 10
def avg_age_teachers : ℚ := 45

-- Define the theorem
theorem average_age_combined :
  let total_people := num_fifth_graders + num_parents + num_teachers
  let total_age := num_fifth_graders * avg_age_fifth_graders +
                   num_parents * avg_age_parents +
                   num_teachers * avg_age_teachers
  total_age / total_people = 27.5454545 := by
  sorry


end NUMINAMATH_CALUDE_average_age_combined_l3257_325705


namespace NUMINAMATH_CALUDE_house_development_l3257_325725

theorem house_development (total houses garage pool neither : ℕ) : 
  total = 70 → 
  garage = 50 → 
  pool = 40 → 
  neither = 15 → 
  ∃ both : ℕ, both = garage + pool - (total - neither) :=
by
  sorry

end NUMINAMATH_CALUDE_house_development_l3257_325725


namespace NUMINAMATH_CALUDE_paramEquations_represent_line_l3257_325763

/-- Parametric equations of a line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The line y = 2x + 1 -/
def linearEquation (x y : ℝ) : Prop := y = 2 * x + 1

/-- The parametric equations x = t - 1 and y = 2t - 1 -/
def paramEquations : ParametricLine :=
  { x := fun t => t - 1
    y := fun t => 2 * t - 1 }

/-- Theorem: The parametric equations represent the line y = 2x + 1 -/
theorem paramEquations_represent_line :
  ∀ t : ℝ, linearEquation (paramEquations.x t) (paramEquations.y t) := by
  sorry


end NUMINAMATH_CALUDE_paramEquations_represent_line_l3257_325763


namespace NUMINAMATH_CALUDE_legs_per_chair_correct_l3257_325721

/-- The number of legs per office chair in Kenzo's company -/
def legs_per_chair : ℕ := 5

/-- The initial number of office chairs -/
def initial_chairs : ℕ := 80

/-- The number of round tables -/
def round_tables : ℕ := 20

/-- The number of legs per round table -/
def legs_per_table : ℕ := 3

/-- The percentage of chairs that remain after damage (as a rational number) -/
def remaining_chair_ratio : ℚ := 3/5

/-- The total number of furniture legs remaining after disposal -/
def total_remaining_legs : ℕ := 300

/-- Theorem stating that the number of legs per chair is correct given the conditions -/
theorem legs_per_chair_correct : 
  (remaining_chair_ratio * initial_chairs : ℚ).num * legs_per_chair + 
  round_tables * legs_per_table = total_remaining_legs :=
by sorry

end NUMINAMATH_CALUDE_legs_per_chair_correct_l3257_325721


namespace NUMINAMATH_CALUDE_orangeade_pricing_l3257_325707

/-- Orangeade pricing problem -/
theorem orangeade_pricing
  (orange_juice : ℝ)  -- Amount of orange juice (same for both days)
  (water_day1 : ℝ)    -- Amount of water on day 1
  (water_day2 : ℝ)    -- Amount of water on day 2
  (price_day1 : ℝ)    -- Price per glass on day 1
  (h1 : water_day1 = orange_juice)        -- Equal amounts of orange juice and water on day 1
  (h2 : water_day2 = 2 * orange_juice)    -- Twice the amount of water on day 2
  (h3 : price_day1 = 0.48)                -- Price per glass on day 1 is $0.48
  (h4 : (orange_juice + water_day1) * price_day1 = 
        (orange_juice + water_day2) * price_day2) -- Same revenue on both days
  : price_day2 = 0.32 :=
by sorry

end NUMINAMATH_CALUDE_orangeade_pricing_l3257_325707


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3257_325726

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (20^3 + 15^4 - 10^5) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (20^3 + 15^4 - 10^5) → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3257_325726


namespace NUMINAMATH_CALUDE_queen_mary_legs_l3257_325713

/-- The total number of legs on the Queen Mary II -/
def total_legs : ℕ := 41

/-- The total number of heads on the ship -/
def total_heads : ℕ := 14

/-- The number of cats on the ship -/
def num_cats : ℕ := 7

/-- The number of legs a cat has -/
def cat_legs : ℕ := 4

/-- The number of legs a normal human has -/
def human_legs : ℕ := 2

/-- The number of legs the captain has -/
def captain_legs : ℕ := 1

/-- Theorem stating the total number of legs on the ship -/
theorem queen_mary_legs : 
  total_legs = 
    (num_cats * cat_legs) + 
    ((total_heads - num_cats - 1) * human_legs) + 
    captain_legs :=
by sorry

end NUMINAMATH_CALUDE_queen_mary_legs_l3257_325713


namespace NUMINAMATH_CALUDE_semicircle_radius_in_isosceles_triangle_l3257_325771

/-- An isosceles triangle with a semicircle inscribed -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The diameter of the semicircle is contained in the base of the triangle -/
  diameter_in_base : radius * 2 ≤ base

/-- The theorem stating the radius of the semicircle in the given isosceles triangle -/
theorem semicircle_radius_in_isosceles_triangle 
  (triangle : IsoscelesTriangleWithSemicircle) 
  (h1 : triangle.base = 20) 
  (h2 : triangle.height = 12) : 
  triangle.radius = 60 / (5 + Real.sqrt 61) :=
sorry

end NUMINAMATH_CALUDE_semicircle_radius_in_isosceles_triangle_l3257_325771


namespace NUMINAMATH_CALUDE_f_properties_l3257_325739

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_properties :
  (∀ x : ℝ, f x ≥ 3 ↔ x ≤ -3/2 ∨ x ≥ 3/2) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ a^2 - a) ↔ a ∈ Set.Icc (-1) 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3257_325739


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_n_minus_one_l3257_325777

theorem binomial_coefficient_n_plus_one_choose_n_minus_one (n : ℕ+) :
  Nat.choose (n + 1) (n - 1) = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_n_minus_one_l3257_325777


namespace NUMINAMATH_CALUDE_circles_shortest_distance_l3257_325720

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 6*y = 8

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop :=
  x^2 + 6*x + y^2 - 2*y = 1

/-- The shortest distance between the two circles -/
def shortest_distance : ℝ := -0.68

/-- Theorem stating that the shortest distance between the two circles is -0.68 -/
theorem circles_shortest_distance :
  ∃ (d : ℝ), d = shortest_distance ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    circle1 x₁ y₁ → circle2 x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ d :=
  sorry

end NUMINAMATH_CALUDE_circles_shortest_distance_l3257_325720


namespace NUMINAMATH_CALUDE_leonardo_chocolate_purchase_l3257_325792

theorem leonardo_chocolate_purchase (chocolate_cost : ℕ) (leonardo_money : ℕ) (borrowed_money : ℕ) : 
  chocolate_cost = 500 ∧ leonardo_money = 400 ∧ borrowed_money = 59 →
  chocolate_cost - (leonardo_money + borrowed_money) = 41 :=
by sorry

end NUMINAMATH_CALUDE_leonardo_chocolate_purchase_l3257_325792


namespace NUMINAMATH_CALUDE_library_schedule_lcm_l3257_325702

theorem library_schedule_lcm : Nat.lcm 5 (Nat.lcm 3 (Nat.lcm 9 8)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_library_schedule_lcm_l3257_325702


namespace NUMINAMATH_CALUDE_largest_percent_error_circle_area_l3257_325716

/-- The largest possible percent error in the computed area of a circle -/
theorem largest_percent_error_circle_area (actual_circumference : ℝ) (max_error_percent : ℝ) :
  actual_circumference = 30 →
  max_error_percent = 15 →
  ∃ (computed_area actual_area : ℝ),
    computed_area ≠ actual_area ∧
    abs ((computed_area - actual_area) / actual_area) ≤ 0.3225 ∧
    ∀ (other_area : ℝ),
      abs ((other_area - actual_area) / actual_area) ≤ abs ((computed_area - actual_area) / actual_area) :=
by sorry

end NUMINAMATH_CALUDE_largest_percent_error_circle_area_l3257_325716


namespace NUMINAMATH_CALUDE_common_ratio_is_negative_half_l3257_325750

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  not_constant : ∃ i j, a i ≠ a j
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  a_3 : a 3 = 5 / 2
  S_3 : (a 1) + (a 2) + (a 3) = 15 / 2

/-- The common ratio of the geometric sequence -/
def common_ratio (seq : GeometricSequence) : ℚ := seq.a 2 / seq.a 1

theorem common_ratio_is_negative_half (seq : GeometricSequence) : 
  common_ratio seq = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_negative_half_l3257_325750


namespace NUMINAMATH_CALUDE_highest_number_on_paper_l3257_325778

theorem highest_number_on_paper (n : ℕ) : 
  (1 : ℚ) / n = 0.01020408163265306 → n = 98 :=
by sorry

end NUMINAMATH_CALUDE_highest_number_on_paper_l3257_325778


namespace NUMINAMATH_CALUDE_even_function_range_l3257_325729

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem even_function_range (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 2, f a b x = f a b (-x)) →
  Set.range (f a b) = Set.Icc (-10) 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_range_l3257_325729


namespace NUMINAMATH_CALUDE_bottle_production_theorem_l3257_325756

/-- Given a number of machines and their production rate, calculate the total bottles produced in a given time -/
def bottlesProduced (numMachines : ℕ) (ratePerMinute : ℕ) (minutes : ℕ) : ℕ :=
  numMachines * ratePerMinute * minutes

/-- The production rate of a single machine -/
def singleMachineRate (totalMachines : ℕ) (totalRate : ℕ) : ℕ :=
  totalRate / totalMachines

theorem bottle_production_theorem (initialMachines : ℕ) (initialRate : ℕ) (newMachines : ℕ) (time : ℕ) :
  initialMachines = 6 →
  initialRate = 270 →
  newMachines = 14 →
  time = 4 →
  bottlesProduced newMachines (singleMachineRate initialMachines initialRate) time = 2520 :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_production_theorem_l3257_325756


namespace NUMINAMATH_CALUDE_solution_set_of_polynomial_equation_l3257_325740

theorem solution_set_of_polynomial_equation :
  let S := {x : ℝ | x = 0 ∨ 
                   x = Real.sqrt ((5 + Real.sqrt 5) / 2) ∨ 
                   x = -Real.sqrt ((5 + Real.sqrt 5) / 2) ∨ 
                   x = Real.sqrt ((5 - Real.sqrt 5) / 2) ∨ 
                   x = -Real.sqrt ((5 - Real.sqrt 5) / 2)}
  ∀ x : ℝ, (5*x - 5*x^3 + x^5 = 0) ↔ x ∈ S :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_polynomial_equation_l3257_325740


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l3257_325733

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l3257_325733


namespace NUMINAMATH_CALUDE_group_size_l3257_325783

theorem group_size (average_increase : ℝ) (weight_difference : ℝ) :
  average_increase = 3.5 →
  weight_difference = 28 →
  weight_difference = average_increase * 8 :=
by sorry

end NUMINAMATH_CALUDE_group_size_l3257_325783


namespace NUMINAMATH_CALUDE_car_wash_goal_proof_l3257_325736

def car_wash_goal (families_10 : ℕ) (amount_10 : ℕ) (families_5 : ℕ) (amount_5 : ℕ) (more_needed : ℕ) : Prop :=
  let earned_10 := families_10 * amount_10
  let earned_5 := families_5 * amount_5
  let total_earned := earned_10 + earned_5
  let goal := total_earned + more_needed
  goal = 150

theorem car_wash_goal_proof :
  car_wash_goal 3 10 15 5 45 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_goal_proof_l3257_325736


namespace NUMINAMATH_CALUDE_indeterminateNatureAndSanity_l3257_325772

-- Define the types for Transylvanians
inductive Transylvanian
| Human
| Vampire

-- Define the mental state
inductive MentalState
| Sane
| Insane

-- Define reliability
def isReliable (t : Transylvanian) (m : MentalState) : Prop :=
  (t = Transylvanian.Human ∧ m = MentalState.Sane) ∨
  (t = Transylvanian.Vampire ∧ m = MentalState.Insane)

-- Define unreliability
def isUnreliable (t : Transylvanian) (m : MentalState) : Prop :=
  (t = Transylvanian.Human ∧ m = MentalState.Insane) ∨
  (t = Transylvanian.Vampire ∧ m = MentalState.Sane)

-- Define the statement function
def statesTrue (t : Transylvanian) (m : MentalState) : Prop :=
  isReliable t m

-- Define the answer to the question "Are you reliable?"
def answersYes (t : Transylvanian) (m : MentalState) : Prop :=
  (isReliable t m ∧ statesTrue t m) ∨ (isUnreliable t m ∧ ¬statesTrue t m)

-- Theorem: It's impossible to determine the nature or sanity of a Transylvanian
-- based on their answer to the question "Are you reliable?"
theorem indeterminateNatureAndSanity (t : Transylvanian) (m : MentalState) :
  answersYes t m → 
  (∃ (t' : Transylvanian) (m' : MentalState), t' ≠ t ∨ m' ≠ m) ∧ answersYes t' m' :=
sorry


end NUMINAMATH_CALUDE_indeterminateNatureAndSanity_l3257_325772


namespace NUMINAMATH_CALUDE_cos_sq_plus_two_sin_double_l3257_325789

theorem cos_sq_plus_two_sin_double (α : Real) (h : Real.tan α = 3 / 4) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_sq_plus_two_sin_double_l3257_325789


namespace NUMINAMATH_CALUDE_existence_of_57_multiple_non_existence_of_58_multiple_l3257_325731

/-- Removes the first digit of a positive integer -/
def removeFirstDigit (n : ℕ) : ℕ := sorry

/-- Checks if a number satisfies the condition A = k * B, where B is A with its first digit removed -/
def satisfiesCondition (A : ℕ) (k : ℕ) : Prop :=
  A = k * removeFirstDigit A

theorem existence_of_57_multiple :
  ∃ A : ℕ, A > 0 ∧ satisfiesCondition A 57 := by sorry

theorem non_existence_of_58_multiple :
  ¬∃ A : ℕ, A > 0 ∧ satisfiesCondition A 58 := by sorry

end NUMINAMATH_CALUDE_existence_of_57_multiple_non_existence_of_58_multiple_l3257_325731


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l3257_325755

theorem angle_sum_in_circle (x : ℝ) : 
  (3 * x + 7 * x + 4 * x + x = 360) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l3257_325755


namespace NUMINAMATH_CALUDE_tangent_line_at_2_l3257_325793

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

-- Theorem statement
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), 
    (∀ x y, y = m*x + b ↔ x - y - 4 = 0) ∧ 
    (m = f' 2) ∧
    (f 2 = m*2 + b) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_l3257_325793


namespace NUMINAMATH_CALUDE_min_value_a5_plus_a6_l3257_325786

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n

-- Define the theorem
theorem min_value_a5_plus_a6 (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6 →
  ∃ (min : ℝ), min = 48 ∧ ∀ (a : ℕ → ℝ),
    is_positive_geometric_sequence a →
    a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6 →
    a 5 + a 6 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_a5_plus_a6_l3257_325786


namespace NUMINAMATH_CALUDE_f_even_implies_increasing_l3257_325708

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

theorem f_even_implies_increasing (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  ∀ a b, 0 < a → a < b → f m a < f m b :=
by sorry

end NUMINAMATH_CALUDE_f_even_implies_increasing_l3257_325708


namespace NUMINAMATH_CALUDE_f_increasing_on_negative_reals_l3257_325784

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 1

-- State the theorem
theorem f_increasing_on_negative_reals (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is an even function
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f m x < f m y) :=  -- f is increasing on (-∞, 0]
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_negative_reals_l3257_325784


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3257_325761

theorem trig_expression_equals_one : 
  let cos30 := Real.cos (30 * π / 180)
  let sin60 := Real.sin (60 * π / 180)
  let sin30 := Real.sin (30 * π / 180)
  let cos60 := Real.cos (60 * π / 180)
  (1 - 1/cos30) * (1 + 1/sin60) * (1 - 1/sin30) * (1 + 1/cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3257_325761


namespace NUMINAMATH_CALUDE_base_subtraction_l3257_325717

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement --/
theorem base_subtraction :
  let base_8_num := to_base_10 [0, 1, 2, 3, 4, 5] 8
  let base_9_num := to_base_10 [2, 3, 4, 5, 6] 9
  base_8_num - base_9_num = 136532 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l3257_325717


namespace NUMINAMATH_CALUDE_jason_initial_cards_l3257_325785

theorem jason_initial_cards (cards_sold : ℕ) (cards_remaining : ℕ) : 
  cards_sold = 224 → cards_remaining = 452 → cards_sold + cards_remaining = 676 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l3257_325785


namespace NUMINAMATH_CALUDE_fifth_item_equals_one_fifteenth_l3257_325788

-- Define the sequence a_n
def a (n : ℕ) : ℚ := 2 / (n^2 + n : ℚ)

-- Theorem statement
theorem fifth_item_equals_one_fifteenth : a 5 = 1/15 := by
  sorry

end NUMINAMATH_CALUDE_fifth_item_equals_one_fifteenth_l3257_325788


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3257_325722

-- Define propositions A, B, and C
variable (A B C : Prop)

-- Define the given conditions
variable (h1 : B → A)
variable (h2 : C → B)
variable (h3 : ¬(B → C))

-- Theorem to prove
theorem sufficient_not_necessary : (C → A) ∧ ¬(A → C) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3257_325722


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3257_325759

/-- Given two parallel vectors a and b in R², where a = (1, 2) and b = (-2, y),
    prove that y must equal -4. -/
theorem parallel_vectors_y_value (y : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, y]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  y = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3257_325759


namespace NUMINAMATH_CALUDE_number_of_subsets_of_intersection_l3257_325782

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {0, 2, 4}

theorem number_of_subsets_of_intersection : Finset.card (Finset.powerset (M ∩ N)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_subsets_of_intersection_l3257_325782


namespace NUMINAMATH_CALUDE_dot_product_OA_OB_is_zero_l3257_325757

theorem dot_product_OA_OB_is_zero (OA OB : ℝ × ℝ) : 
  OA = (1, -3) →
  ‖OA‖ = ‖OB‖ →
  ‖OA - OB‖ = 2 * Real.sqrt 5 →
  OA • OB = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_OA_OB_is_zero_l3257_325757


namespace NUMINAMATH_CALUDE_combined_instruments_count_l3257_325752

/-- Represents the number of musical instruments owned by a person -/
structure Instruments where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- Calculates the total number of instruments -/
def totalInstruments (i : Instruments) : ℕ :=
  i.flutes + i.horns + i.harps

/-- Charlie's instruments -/
def charlie : Instruments :=
  { flutes := 1, horns := 2, harps := 1 }

/-- Carli's instruments -/
def carli : Instruments :=
  { flutes := 2 * charlie.flutes,
    horns := charlie.horns / 2,
    harps := 0 }

theorem combined_instruments_count :
  totalInstruments charlie + totalInstruments carli = 7 := by
  sorry

end NUMINAMATH_CALUDE_combined_instruments_count_l3257_325752


namespace NUMINAMATH_CALUDE_oliver_bath_frequency_l3257_325730

def bucket_capacity : ℕ := 120
def buckets_to_fill : ℕ := 14
def buckets_removed : ℕ := 3
def weekly_water_usage : ℕ := 9240

theorem oliver_bath_frequency :
  let full_tub := bucket_capacity * buckets_to_fill
  let water_removed := bucket_capacity * buckets_removed
  let water_per_bath := full_tub - water_removed
  weekly_water_usage / water_per_bath = 7 := by sorry

end NUMINAMATH_CALUDE_oliver_bath_frequency_l3257_325730


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3257_325754

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 540) : 
  1.2 * L * (0.8 * W) = 518.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3257_325754


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3257_325747

theorem cos_alpha_value (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.cos (α + Real.pi / 3) = -2/3) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 := by sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3257_325747


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3257_325741

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3257_325741


namespace NUMINAMATH_CALUDE_max_non_managers_dept_A_l3257_325753

/-- Represents a department in the company -/
inductive Department
| A
| B
| C

/-- Represents the gender of an employee -/
inductive Gender
| Male
| Female

/-- Represents the status of a manager -/
inductive ManagerStatus
| Active
| OnVacation

/-- Represents the type of non-manager employee -/
inductive NonManagerType
| FullTime
| PartTime

/-- The company structure and policies -/
structure Company where
  /-- The ratio of managers to non-managers must be greater than this for all departments -/
  baseRatio : Rat
  /-- Department A's specific ratio requirement -/
  deptARatio : Rat
  /-- Department B's specific ratio requirement -/
  deptBRatio : Rat
  /-- The minimum gender ratio (male:female) for non-managers -/
  genderRatio : Rat

/-- Represents the workforce of a department -/
structure DepartmentWorkforce where
  department : Department
  totalManagers : Nat
  activeManagers : Nat
  nonManagersMale : Nat
  nonManagersFemale : Nat
  partTimeNonManagers : Nat

/-- Main theorem to prove -/
theorem max_non_managers_dept_A (c : Company) (dA : DepartmentWorkforce) :
  c.baseRatio = 7/32 ∧
  c.deptARatio = 9/33 ∧
  c.deptBRatio = 8/34 ∧
  c.genderRatio = 1/2 ∧
  dA.department = Department.A ∧
  dA.totalManagers = 8 ∧
  dA.activeManagers = 4 →
  dA.nonManagersMale + dA.nonManagersFemale + dA.partTimeNonManagers / 2 ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_non_managers_dept_A_l3257_325753


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l3257_325796

theorem last_two_digits_sum (n : ℕ) : (9^n + 11^n) % 100 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l3257_325796


namespace NUMINAMATH_CALUDE_giyoons_chocolates_l3257_325710

theorem giyoons_chocolates (initial_friends : ℕ) (absent_friends : ℕ) (extra_per_person : ℕ) (leftover : ℕ) :
  initial_friends = 8 →
  absent_friends = 2 →
  extra_per_person = 1 →
  leftover = 4 →
  ∃ (total_chocolates : ℕ),
    total_chocolates = (initial_friends - absent_friends) * ((total_chocolates / initial_friends) + extra_per_person) + leftover ∧
    total_chocolates = 40 :=
by sorry

end NUMINAMATH_CALUDE_giyoons_chocolates_l3257_325710


namespace NUMINAMATH_CALUDE_prime_pairs_congruence_l3257_325704

theorem prime_pairs_congruence (p q : ℕ) : 
  Prime p ∧ Prime q →
  (∀ x : ℤ, x^(3*p*q) ≡ x [ZMOD (3*p*q)]) →
  ((p = 11 ∧ q = 17) ∨ (p = 17 ∧ q = 11)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_congruence_l3257_325704


namespace NUMINAMATH_CALUDE_root_transformation_l3257_325766

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - r₁^2 + 3*r₁ - 7 = 0) ∧ 
  (r₂^3 - r₂^2 + 3*r₂ - 7 = 0) ∧ 
  (r₃^3 - r₃^2 + 3*r₃ - 7 = 0) →
  ((3*r₁)^3 - 3*(3*r₁)^2 + 27*(3*r₁) - 189 = 0) ∧ 
  ((3*r₂)^3 - 3*(3*r₂)^2 + 27*(3*r₂) - 189 = 0) ∧ 
  ((3*r₃)^3 - 3*(3*r₃)^2 + 27*(3*r₃) - 189 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l3257_325766


namespace NUMINAMATH_CALUDE_triangle_existence_l3257_325769

/-- Represents a triangle with side lengths and angles -/
structure Triangle where
  a : ℝ  -- base length
  b : ℝ  -- one side length
  c : ℝ  -- other side length
  α : ℝ  -- angle opposite to side a
  β : ℝ  -- angle opposite to side b
  γ : ℝ  -- angle opposite to side c

/-- The existence of a triangle with given properties -/
theorem triangle_existence 
  (a : ℝ) 
  (bc_sum : ℝ) 
  (Δθ : ℝ) 
  (h_a_pos : a > 0) 
  (h_bc_sum_pos : bc_sum > 0) 
  (h_Δθ_range : 0 < Δθ ∧ Δθ < π) :
  ∃ (t : Triangle), 
    t.a = a ∧ 
    t.b + t.c = bc_sum ∧ 
    |t.β - t.γ| = Δθ ∧
    t.α + t.β + t.γ = π :=
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l3257_325769


namespace NUMINAMATH_CALUDE_knight_seating_probability_correct_l3257_325719

/-- The probability of three knights seated at a round table with n chairs
    having empty chairs on both sides of each knight. -/
def knight_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2))
  else
    0

/-- Theorem: The probability of three knights seated at a round table with n chairs (n ≥ 6)
    having empty chairs on both sides of each knight is (n-4)(n-5) / ((n-1)(n-2)). -/
theorem knight_seating_probability_correct (n : ℕ) (h : n ≥ 6) :
  knight_seating_probability n = (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_knight_seating_probability_correct_l3257_325719


namespace NUMINAMATH_CALUDE_students_with_cat_and_dog_l3257_325791

theorem students_with_cat_and_dog (total : ℕ) (cat : ℕ) (dog : ℕ) (neither : ℕ) 
  (h1 : total = 28)
  (h2 : cat = 17)
  (h3 : dog = 10)
  (h4 : neither = 5)
  : ∃ both : ℕ, both = cat + dog - (total - neither) :=
by
  sorry

end NUMINAMATH_CALUDE_students_with_cat_and_dog_l3257_325791


namespace NUMINAMATH_CALUDE_check_tianning_pairs_find_x_for_negative_five_evaluate_expression_for_tianning_pair_l3257_325758

-- Define Tianning pair
def is_tianning_pair (a b : ℝ) : Prop := a + b = a * b

-- Theorem 1: Checking specific pairs
theorem check_tianning_pairs :
  is_tianning_pair 3 1.5 ∧
  is_tianning_pair (-1/2) (1/3) ∧
  ¬ is_tianning_pair (3/4) 1 :=
sorry

-- Theorem 2: Finding x for (-5, x)
theorem find_x_for_negative_five :
  ∃ x, is_tianning_pair (-5) x ∧ x = 5/6 :=
sorry

-- Theorem 3: Evaluating expression for any Tianning pair
theorem evaluate_expression_for_tianning_pair (m n : ℝ) :
  is_tianning_pair m n →
  4*(m*n+m-2*(m*n-3))-2*(3*m^2-2*n)+6*m^2 = 24 :=
sorry

end NUMINAMATH_CALUDE_check_tianning_pairs_find_x_for_negative_five_evaluate_expression_for_tianning_pair_l3257_325758


namespace NUMINAMATH_CALUDE_luka_age_when_max_born_l3257_325706

/-- Proves Luka's age when Max was born -/
theorem luka_age_when_max_born (luka_aubrey_age_diff : ℕ) 
  (aubrey_age_at_max_6 : ℕ) (max_age_at_aubrey_8 : ℕ) :
  luka_aubrey_age_diff = 2 →
  aubrey_age_at_max_6 = 8 →
  max_age_at_aubrey_8 = 6 →
  aubrey_age_at_max_6 - max_age_at_aubrey_8 + luka_aubrey_age_diff = 4 :=
by sorry

end NUMINAMATH_CALUDE_luka_age_when_max_born_l3257_325706


namespace NUMINAMATH_CALUDE_blue_parrots_count_l3257_325732

theorem blue_parrots_count (total : ℕ) (green_fraction : ℚ) : 
  total = 160 → 
  green_fraction = 5/8 → 
  (1 - green_fraction) * total = 60 := by
sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l3257_325732


namespace NUMINAMATH_CALUDE_base_10_to_base_3_l3257_325787

theorem base_10_to_base_3 : 
  (2 * 3^6 + 0 * 3^5 + 0 * 3^4 + 1 * 3^3 + 1 * 3^2 + 2 * 3^1 + 2 * 3^0) = 1589 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_3_l3257_325787


namespace NUMINAMATH_CALUDE_compute_expression_l3257_325797

theorem compute_expression : 3 * 3^4 - 27^65 / 27^63 = -486 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3257_325797


namespace NUMINAMATH_CALUDE_james_coins_value_l3257_325723

theorem james_coins_value (p n : ℕ) : 
  p + n = 15 →
  p - 1 = n →
  p * 1 + n * 5 = 43 :=
by sorry

end NUMINAMATH_CALUDE_james_coins_value_l3257_325723


namespace NUMINAMATH_CALUDE_sequence_inequality_l3257_325798

theorem sequence_inequality (n : ℕ) (a : ℕ → ℚ) (h1 : a 0 = 1/2) 
  (h2 : ∀ k, k < n → a (k + 1) = a k + (1/n) * (a k)^2) :
  1 - 1/n < a n ∧ a n < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3257_325798


namespace NUMINAMATH_CALUDE_sqrt_2023_irrational_not_perfect_square_2023_l3257_325712

theorem sqrt_2023_irrational : Irrational (Real.sqrt 2023) := by sorry

theorem not_perfect_square_2023 : ¬ ∃ n : ℕ, n ^ 2 = 2023 := by sorry

end NUMINAMATH_CALUDE_sqrt_2023_irrational_not_perfect_square_2023_l3257_325712


namespace NUMINAMATH_CALUDE_expression_equality_l3257_325714

theorem expression_equality (x : ℝ) : 3 * x * (21 - (x + 3) * x - 3) = 54 * x - 3 * x^3 + 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3257_325714


namespace NUMINAMATH_CALUDE_election_fourth_place_votes_l3257_325745

theorem election_fourth_place_votes :
  ∀ (total_votes : ℕ) (winner_votes : ℕ) (second_place_diff : ℕ) (third_place_diff : ℕ) (fourth_place_diff : ℕ),
    total_votes = 979 →
    winner_votes = second_place_diff + (winner_votes - second_place_diff) →
    winner_votes = third_place_diff + (winner_votes - third_place_diff) →
    winner_votes = fourth_place_diff + (winner_votes - fourth_place_diff) →
    second_place_diff = 53 →
    third_place_diff = 79 →
    fourth_place_diff = 105 →
    total_votes = winner_votes + (winner_votes - second_place_diff) + (winner_votes - third_place_diff) + (winner_votes - fourth_place_diff) →
    winner_votes - fourth_place_diff = 199 :=
by sorry

end NUMINAMATH_CALUDE_election_fourth_place_votes_l3257_325745


namespace NUMINAMATH_CALUDE_fraction_equality_l3257_325799

theorem fraction_equality (a b : ℝ) (h1 : a = (2/3) * b) (h2 : b ≠ 0) : 
  (9*a + 8*b) / (6*a) = 7/2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3257_325799


namespace NUMINAMATH_CALUDE_wall_bricks_count_l3257_325744

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 1800

/-- Time taken by the first bricklayer to build the wall alone -/
def time_bricklayer1 : ℕ := 8

/-- Time taken by the second bricklayer to build the wall alone -/
def time_bricklayer2 : ℕ := 12

/-- Reduction in combined output when working together -/
def output_reduction : ℕ := 15

/-- Time taken to complete the wall when working together -/
def time_together : ℕ := 5

theorem wall_bricks_count :
  (time_together : ℝ) * ((total_bricks / time_bricklayer1 : ℝ) +
  (total_bricks / time_bricklayer2 : ℝ) - output_reduction) = total_bricks := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l3257_325744


namespace NUMINAMATH_CALUDE_aprons_to_sew_tomorrow_l3257_325700

def total_aprons : ℕ := 150
def aprons_before_today : ℕ := 13
def aprons_today : ℕ := 3 * aprons_before_today

def aprons_sewn_so_far : ℕ := aprons_before_today + aprons_today
def remaining_aprons : ℕ := total_aprons - aprons_sewn_so_far
def aprons_tomorrow : ℕ := remaining_aprons / 2

theorem aprons_to_sew_tomorrow : aprons_tomorrow = 49 := by
  sorry

end NUMINAMATH_CALUDE_aprons_to_sew_tomorrow_l3257_325700


namespace NUMINAMATH_CALUDE_largest_integer_square_4_digits_base8_l3257_325770

/-- The largest integer whose square has exactly 4 digits in base 8 -/
def N : ℕ := 63

/-- Conversion of a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem largest_integer_square_4_digits_base8 :
  (N^2 ≥ 8^3) ∧ (N^2 < 8^4) ∧ (∀ m : ℕ, m > N → m^2 ≥ 8^4) ∧ (toBase8 N = [7, 7]) := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_square_4_digits_base8_l3257_325770


namespace NUMINAMATH_CALUDE_not_in_first_quadrant_l3257_325775

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def FirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Theorem: If mn ≤ 0, then point A(m,n) cannot be in the first quadrant -/
theorem not_in_first_quadrant (m n : ℝ) (h : m * n ≤ 0) :
  ¬FirstQuadrant ⟨m, n⟩ := by
  sorry


end NUMINAMATH_CALUDE_not_in_first_quadrant_l3257_325775


namespace NUMINAMATH_CALUDE_g_equals_h_intersection_at_most_one_point_l3257_325765

-- Define the functions g and h
def g (x : ℝ) : ℝ := 2 * x - 1
def h (t : ℝ) : ℝ := 2 * t - 1

-- Theorem 1: g and h are the same function
theorem g_equals_h : g = h := by sorry

-- Theorem 2: For any function f, the intersection of y = f(x) and x = 2 has at most one point
theorem intersection_at_most_one_point (f : ℝ → ℝ) :
  ∃! y, f 2 = y := by sorry

end NUMINAMATH_CALUDE_g_equals_h_intersection_at_most_one_point_l3257_325765


namespace NUMINAMATH_CALUDE_decimal_difference_l3257_325760

/-- The value of the repeating decimal 0.717171... -/
def repeating_decimal : ℚ := 71 / 99

/-- The value of the terminating decimal 0.71 -/
def terminating_decimal : ℚ := 71 / 100

/-- Theorem stating that the difference between 0.717171... and 0.71 is 71/9900 -/
theorem decimal_difference : repeating_decimal - terminating_decimal = 71 / 9900 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l3257_325760


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l3257_325762

theorem max_value_x_plus_y (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 6 * y) :
  ∃ (max : ℝ), ∀ (x' y' : ℝ), 2 * x'^2 + 3 * y'^2 = 6 * y' → x' + y' ≤ max ∧ max = 1 + Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l3257_325762


namespace NUMINAMATH_CALUDE_pet_shelter_adoption_time_l3257_325779

/-- Given an initial number of puppies, additional puppies, and a daily adoption rate,
    calculate the number of days required to adopt all puppies. -/
def days_to_adopt (initial : ℕ) (additional : ℕ) (adoption_rate : ℕ) : ℕ :=
  (initial + additional) / adoption_rate

/-- Theorem: For the given problem, it takes 2 days to adopt all puppies. -/
theorem pet_shelter_adoption_time : days_to_adopt 3 3 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_shelter_adoption_time_l3257_325779


namespace NUMINAMATH_CALUDE_triangle_midpoint_x_coordinate_sum_l3257_325735

theorem triangle_midpoint_x_coordinate_sum (a b c : ℝ) :
  let vertex_sum := a + b + c
  let midpoint_sum := (a + b) / 2 + (b + c) / 2 + (c + a) / 2
  midpoint_sum = vertex_sum := by
sorry

end NUMINAMATH_CALUDE_triangle_midpoint_x_coordinate_sum_l3257_325735


namespace NUMINAMATH_CALUDE_multiple_solutions_exist_l3257_325742

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem multiple_solutions_exist :
  ∃ (c₁ c₂ : ℝ), c₁ ≠ c₂ ∧ f (f (f (f c₁))) = 2 ∧ f (f (f (f c₂))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_multiple_solutions_exist_l3257_325742


namespace NUMINAMATH_CALUDE_x_squared_when_y_is_4_l3257_325718

-- Define the inverse variation relationship between x² and y³
def inverse_variation (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x^2 * y^3 = k

-- State the theorem
theorem x_squared_when_y_is_4
  (h1 : ∀ x y, inverse_variation x y)
  (h2 : inverse_variation 10 2) :
  ∃ x : ℝ, inverse_variation x 4 ∧ x^2 = 12.5 := by
sorry


end NUMINAMATH_CALUDE_x_squared_when_y_is_4_l3257_325718


namespace NUMINAMATH_CALUDE_inequality_proof_l3257_325776

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ 1/a + 1/b + 1/c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3257_325776


namespace NUMINAMATH_CALUDE_tony_fever_threshold_l3257_325764

/-- Calculates how many degrees above the fever threshold a person's temperature is -/
def degrees_above_fever_threshold (normal_temp fever_threshold temp_increase : ℝ) : ℝ :=
  (normal_temp + temp_increase) - fever_threshold

/-- Proves that Tony's temperature is 5 degrees above the fever threshold -/
theorem tony_fever_threshold :
  degrees_above_fever_threshold 95 100 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tony_fever_threshold_l3257_325764


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3257_325734

theorem two_digit_number_property (N : ℕ) : 
  (N ≥ 10 ∧ N ≤ 99) →
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →
  (N = 32 ∨ N = 64 ∨ N = 96) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3257_325734


namespace NUMINAMATH_CALUDE_warren_guests_calculation_l3257_325749

/-- The number of tables Warren has -/
def num_tables : ℝ := 252.0

/-- The number of guests each table can hold -/
def guests_per_table : ℝ := 4.0

/-- The total number of guests Warren can accommodate -/
def total_guests : ℝ := num_tables * guests_per_table

theorem warren_guests_calculation : total_guests = 1008 := by
  sorry

end NUMINAMATH_CALUDE_warren_guests_calculation_l3257_325749


namespace NUMINAMATH_CALUDE_frustum_volume_ratio_l3257_325709

theorem frustum_volume_ratio (h₁ h₂ : ℝ) (A₁ A₂ : ℝ) (V₁ V₂ : ℝ) :
  h₁ / h₂ = 3 / 5 →
  A₁ / A₂ = 9 / 25 →
  V₁ = (1 / 3) * h₁ * A₁ →
  V₂ = (1 / 3) * h₂ * A₂ →
  V₁ / (V₂ - V₁) = 27 / 71 :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_ratio_l3257_325709


namespace NUMINAMATH_CALUDE_stream_rate_proof_l3257_325748

/-- Proves that the rate of a stream is 5 km/hr given the conditions of a boat's travel --/
theorem stream_rate_proof (boat_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  boat_speed = 16 →
  distance = 147 →
  time = 7 →
  (distance / time) - boat_speed = 5 := by
  sorry

#check stream_rate_proof

end NUMINAMATH_CALUDE_stream_rate_proof_l3257_325748


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3257_325701

/-- Given two parallel lines with a distance of 2 between them, where one line has the equation 5x - 12y + 6 = 0, prove that the equation of the other line is either 5x - 12y + 32 = 0 or 5x - 12y - 20 = 0 -/
theorem parallel_line_equation (x y : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 5 * x - 12 * y + 6 = 0
  let l : ℝ → ℝ → Prop := λ x y ↦ 5 * x - 12 * y + 32 = 0 ∨ 5 * x - 12 * y - 20 = 0
  let parallel : Prop := ∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, l₁ x y ↔ l x y
  let distance : ℝ := 2
  parallel → (∀ x y, l x y ↔ (5 * x - 12 * y + 32 = 0 ∨ 5 * x - 12 * y - 20 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3257_325701


namespace NUMINAMATH_CALUDE_unique_solution_l3257_325724

theorem unique_solution : ∃! x : ℝ, ((52 + x) * 3 - 60) / 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3257_325724


namespace NUMINAMATH_CALUDE_fraction_simplification_l3257_325751

theorem fraction_simplification (x : ℝ) (h : x = 3) : 
  (x^8 + 16*x^4 + 64 + 4*x^2) / (x^4 + 8) = 89 + 36/89 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3257_325751


namespace NUMINAMATH_CALUDE_no_nontrivial_solution_for_4n_plus_3_prime_l3257_325737

theorem no_nontrivial_solution_for_4n_plus_3_prime (a : ℕ) (x y z : ℤ) :
  Prime a →
  (∃ n : ℕ, a = 4 * n + 3) →
  x^2 + y^2 = a * z^2 →
  x = 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_nontrivial_solution_for_4n_plus_3_prime_l3257_325737


namespace NUMINAMATH_CALUDE_fraction_of_sum_l3257_325711

theorem fraction_of_sum (m n : ℝ) (a b c : ℝ) 
  (h1 : a = (b + c) / m)
  (h2 : b = (c + a) / n)
  (h3 : m ≠ 0)
  (h4 : n ≠ 0) :
  (m * n ≠ 1 → c / (a + b) = (m * n - 1) / (m + n + 2)) ∧
  (m = -1 ∧ n = -1 → c / (a + b) = -1) :=
sorry

end NUMINAMATH_CALUDE_fraction_of_sum_l3257_325711


namespace NUMINAMATH_CALUDE_first_candidate_vote_percentage_l3257_325794

/-- Proves that the first candidate received 80% of the votes in an election with two candidates -/
theorem first_candidate_vote_percentage
  (total_votes : ℕ)
  (second_candidate_votes : ℕ)
  (h_total : total_votes = 2400)
  (h_second : second_candidate_votes = 480) :
  (total_votes - second_candidate_votes : ℚ) / total_votes * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_vote_percentage_l3257_325794


namespace NUMINAMATH_CALUDE_nature_reserve_count_l3257_325738

theorem nature_reserve_count (total_heads total_legs : ℕ) 
  (h1 : total_heads = 300)
  (h2 : total_legs = 688) : ∃ (birds mammals reptiles : ℕ),
  birds + mammals + reptiles = total_heads ∧
  2 * birds + 4 * mammals + 6 * reptiles = total_legs ∧
  birds = 234 := by
  sorry

end NUMINAMATH_CALUDE_nature_reserve_count_l3257_325738


namespace NUMINAMATH_CALUDE_fractional_equation_root_l3257_325715

theorem fractional_equation_root (m : ℝ) : 
  (∃ x : ℝ, x ≠ 4 ∧ (3 / (x - 4) + (x + m) / (4 - x) = 1)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l3257_325715


namespace NUMINAMATH_CALUDE_playground_boys_count_l3257_325790

/-- Given a playground with children, prove that the number of boys is 44 -/
theorem playground_boys_count (total_children : ℕ) (girls : ℕ) (boys : ℕ) : 
  total_children = 97 → girls = 53 → total_children = girls + boys → boys = 44 := by
sorry

end NUMINAMATH_CALUDE_playground_boys_count_l3257_325790


namespace NUMINAMATH_CALUDE_jacket_cost_l3257_325795

theorem jacket_cost (total_spent : ℚ) (shorts_cost : ℚ) (shirt_cost : ℚ) 
  (h1 : total_spent = 33.56)
  (h2 : shorts_cost = 13.99)
  (h3 : shirt_cost = 12.14) :
  total_spent - (shorts_cost + shirt_cost) = 7.43 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_l3257_325795


namespace NUMINAMATH_CALUDE_sector_inscribed_circle_area_ratio_l3257_325728

theorem sector_inscribed_circle_area_ratio (α : Real) :
  let R := 1  -- We can set R to 1 without loss of generality
  let r := (R * Real.sin (α / 2)) / (1 + Real.sin (α / 2))
  let sector_area := (1 / 2) * R^2 * α
  let inscribed_circle_area := Real.pi * r^2
  sector_area / inscribed_circle_area = (2 * α * (Real.cos (Real.pi / 4 - α / 4))^2) / (Real.pi * (Real.sin (α / 2))^2) :=
by sorry

end NUMINAMATH_CALUDE_sector_inscribed_circle_area_ratio_l3257_325728


namespace NUMINAMATH_CALUDE_class_composition_theorem_l3257_325746

/-- Represents the number of students in a class with specific friendship conditions -/
structure ClassComposition where
  boys : ℕ
  girls : ℕ
  total_children : ℕ
  desks : ℕ

/-- Checks if the class composition satisfies the given conditions -/
def is_valid_composition (c : ClassComposition) : Prop :=
  c.boys * 2 = c.girls * 3 ∧
  c.boys + c.girls = c.total_children ∧
  c.total_children = 31 ∧
  c.desks = 19

/-- Theorem stating that the only valid class composition has 35 students -/
theorem class_composition_theorem :
  ∀ c : ClassComposition, is_valid_composition c → c.boys + c.girls = 35 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_theorem_l3257_325746


namespace NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l3257_325781

/-- The sum of the first n odd positive integers starting from a given odd number -/
def sum_odd_integers (start : ℕ) (n : ℕ) : ℕ :=
  n * (2 * start + n - 1)

/-- The 15th odd positive integer starting from 5 -/
def last_term : ℕ := 5 + 2 * (15 - 1)

theorem sum_first_15_odd_from_5 :
  sum_odd_integers 5 15 = 285 := by sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l3257_325781


namespace NUMINAMATH_CALUDE_rectangle_area_l3257_325780

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 166) : L * B = 1590 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3257_325780


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_ge_sum_l3257_325767

theorem gcd_lcm_sum_ge_sum (a b : ℕ+) : Nat.gcd a b + Nat.lcm a b ≥ a + b := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_ge_sum_l3257_325767


namespace NUMINAMATH_CALUDE_mango_selling_price_l3257_325773

theorem mango_selling_price 
  (loss_percentage : ℝ) 
  (profit_percentage : ℝ) 
  (profit_price : ℝ) :
  loss_percentage = 20 →
  profit_percentage = 5 →
  profit_price = 6.5625 →
  ∃ (actual_price : ℝ), 
    actual_price = (1 - loss_percentage / 100) * (profit_price / (1 + profit_percentage / 100)) ∧
    actual_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_mango_selling_price_l3257_325773


namespace NUMINAMATH_CALUDE_derivative_at_one_l3257_325768

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3257_325768


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3257_325774

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3257_325774


namespace NUMINAMATH_CALUDE_binary_multiplication_l3257_325743

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec toBinary (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
    toBinary n

theorem binary_multiplication :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, true, true, false, false, true]  -- 1001111₂
  binaryToNat a * binaryToNat b = binaryToNat c := by
sorry

end NUMINAMATH_CALUDE_binary_multiplication_l3257_325743


namespace NUMINAMATH_CALUDE_two_blue_gumballs_probability_l3257_325727

/-- The probability of drawing a pink gumball from the jar -/
def prob_pink : ℝ := 0.5714285714285714

/-- The probability of drawing a blue gumball from the jar -/
def prob_blue : ℝ := 1 - prob_pink

/-- The probability of drawing two blue gumballs in a row -/
def prob_two_blue : ℝ := prob_blue * prob_blue

theorem two_blue_gumballs_probability :
  prob_two_blue = 0.1836734693877551 := by sorry

end NUMINAMATH_CALUDE_two_blue_gumballs_probability_l3257_325727
