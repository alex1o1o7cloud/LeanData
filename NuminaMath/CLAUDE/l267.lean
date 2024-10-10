import Mathlib

namespace quadratic_equation_solution_l267_26706

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 5 * x - 3
  ∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = 1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by
  sorry

end quadratic_equation_solution_l267_26706


namespace arithmetic_sequence_middle_term_l267_26752

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) (h_sum : a 1 + a 19 = -18) :
  a 10 = -9 :=
sorry

end arithmetic_sequence_middle_term_l267_26752


namespace max_value_theorem_l267_26798

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - |x - a|

-- State the theorem
theorem max_value_theorem (a b : ℝ) :
  (-1 ≤ a) →
  (a ≤ 1) →
  (∀ x ∈ Set.Icc 1 3, f a x + b * x ≤ 0) →
  (a^2 + 3 * b ≤ 10) ∧ 
  (∃ a₀ b₀, (-1 ≤ a₀) ∧ (a₀ ≤ 1) ∧ 
   (∀ x ∈ Set.Icc 1 3, f a₀ x + b₀ * x ≤ 0) ∧ 
   (a₀^2 + 3 * b₀ = 10)) :=
by sorry

end max_value_theorem_l267_26798


namespace least_subtraction_l267_26751

theorem least_subtraction (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((997 - y) % 5 = 3 ∧ (997 - y) % 9 = 3 ∧ (997 - y) % 11 = 3)) →
  (997 - x) % 5 = 3 ∧ (997 - x) % 9 = 3 ∧ (997 - x) % 11 = 3 →
  x = 4 :=
by sorry

end least_subtraction_l267_26751


namespace total_age_of_couple_l267_26738

def bride_age : ℕ := 102
def age_difference : ℕ := 19

theorem total_age_of_couple : 
  bride_age + (bride_age - age_difference) = 185 := by sorry

end total_age_of_couple_l267_26738


namespace min_total_cost_l267_26768

/-- Represents the transportation problem with two warehouses and two construction sites -/
structure TransportationProblem where
  warehouseA_capacity : ℝ
  warehouseB_capacity : ℝ
  siteA_demand : ℝ
  siteB_demand : ℝ
  costA_to_A : ℝ
  costA_to_B : ℝ
  costB_to_A : ℝ
  costB_to_B : ℝ

/-- The specific transportation problem instance -/
def problem : TransportationProblem :=
  { warehouseA_capacity := 800
  , warehouseB_capacity := 1200
  , siteA_demand := 1300
  , siteB_demand := 700
  , costA_to_A := 12
  , costA_to_B := 15
  , costB_to_A := 10
  , costB_to_B := 18
  }

/-- The cost reduction from Warehouse A to Site A -/
def cost_reduction (a : ℝ) : Prop := 2 ≤ a ∧ a ≤ 6

/-- The amount transported from Warehouse A to Site A -/
def transport_amount (x : ℝ) : Prop := 100 ≤ x ∧ x ≤ 800

/-- The theorem stating the minimum total transportation cost after cost reduction -/
theorem min_total_cost (p : TransportationProblem) (a : ℝ) (x : ℝ) 
  (h1 : p = problem) (h2 : cost_reduction a) (h3 : transport_amount x) : 
  ∃ y : ℝ, y = 22400 ∧ ∀ z : ℝ, z ≥ y := by
  sorry

end min_total_cost_l267_26768


namespace inverse_variation_problem_l267_26748

/-- Given that quantities a and b vary inversely, this function represents their relationship -/
def inverse_variation (k : ℝ) (a b : ℝ) : Prop := a * b = k

theorem inverse_variation_problem (k : ℝ) :
  inverse_variation k 800 0.5 →
  inverse_variation k 1600 0.25 :=
sorry

end inverse_variation_problem_l267_26748


namespace inequality_proof_l267_26707

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := by sorry

end inequality_proof_l267_26707


namespace quadratic_increasing_condition_l267_26793

/-- A quadratic function of the form y = x^2 + (1-m)x + 1 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + (1-m)*x + 1

/-- The derivative of the quadratic function -/
def quadratic_derivative (m : ℝ) (x : ℝ) : ℝ := 2*x + (1-m)

theorem quadratic_increasing_condition (m : ℝ) :
  (∀ x > 1, quadratic_derivative m x > 0) ↔ m ≤ 3 :=
sorry

end quadratic_increasing_condition_l267_26793


namespace line_plane_relationship_l267_26796

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the contained relation (line contained in a plane)
variable (contained_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_relationship 
  (m n : Line) (α : Plane) 
  (h1 : perp_line_plane m α) 
  (h2 : perp_line_line m n) : 
  parallel_line_plane n α ∨ contained_line_plane n α :=
sorry

end line_plane_relationship_l267_26796


namespace clock_divisibility_impossible_l267_26771

theorem clock_divisibility_impossible (a b : ℕ) : 
  0 < a → a ≤ 12 → b < 60 → 
  ¬ (∃ k : ℕ, (120 * a + 2 * b) = k * (100 * a + b)) := by
  sorry

end clock_divisibility_impossible_l267_26771


namespace min_value_cos_squared_minus_sin_squared_l267_26787

theorem min_value_cos_squared_minus_sin_squared :
  ∃ (m : ℝ), (∀ x, m ≤ (Real.cos (x/2))^2 - (Real.sin (x/2))^2) ∧ 
  (∃ x₀, m = (Real.cos (x₀/2))^2 - (Real.sin (x₀/2))^2) ∧
  m = -1 := by
  sorry

end min_value_cos_squared_minus_sin_squared_l267_26787


namespace polynomial_remainder_l267_26785

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^3 - 20 * x^2 + 28 * x - 30) % (4 * x - 8) = 10 := by
sorry

end polynomial_remainder_l267_26785


namespace ratio_to_two_l267_26780

theorem ratio_to_two (x : ℝ) : (x / 2 = 150 / 1) → x = 300 := by
  sorry

end ratio_to_two_l267_26780


namespace carries_payment_l267_26766

/-- Calculate Carrie's payment for clothes shopping --/
theorem carries_payment (shirt_quantity : ℕ) (pants_quantity : ℕ) (jacket_quantity : ℕ) 
  (skirt_quantity : ℕ) (shoes_quantity : ℕ) (shirt_price : ℚ) (pants_price : ℚ) 
  (jacket_price : ℚ) (skirt_price : ℚ) (shoes_price : ℚ) (shirt_discount : ℚ) 
  (jacket_discount : ℚ) (skirt_discount : ℚ) (mom_payment_ratio : ℚ) :
  shirt_quantity = 8 →
  pants_quantity = 4 →
  jacket_quantity = 4 →
  skirt_quantity = 3 →
  shoes_quantity = 2 →
  shirt_price = 12 →
  pants_price = 25 →
  jacket_price = 75 →
  skirt_price = 30 →
  shoes_price = 50 →
  shirt_discount = 0.2 →
  jacket_discount = 0.2 →
  skirt_discount = 0.1 →
  mom_payment_ratio = 2/3 →
  let total_cost := 
    (shirt_quantity : ℚ) * shirt_price * (1 - shirt_discount) +
    (pants_quantity : ℚ) * pants_price +
    (jacket_quantity : ℚ) * jacket_price * (1 - jacket_discount) +
    (skirt_quantity : ℚ) * skirt_price * (1 - skirt_discount) +
    (shoes_quantity : ℚ) * shoes_price
  (1 - mom_payment_ratio) * total_cost = 199.27 := by
  sorry

end carries_payment_l267_26766


namespace bug_probability_l267_26747

def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - P n)

theorem bug_probability : P 12 = 683/2048 := by sorry

end bug_probability_l267_26747


namespace num_selections_with_A_or_B_l267_26720

/-- The number of key projects -/
def num_key_projects : ℕ := 4

/-- The number of general projects -/
def num_general_projects : ℕ := 6

/-- The number of key projects to be selected -/
def select_key : ℕ := 2

/-- The number of general projects to be selected -/
def select_general : ℕ := 2

/-- Theorem stating the number of selection methods with at least one of A or B selected -/
theorem num_selections_with_A_or_B : 
  (Nat.choose (num_key_projects - 1) (select_key - 1) * Nat.choose (num_general_projects - 1) select_general) +
  (Nat.choose (num_key_projects - 1) select_key * Nat.choose (num_general_projects - 1) (select_general - 1)) +
  (Nat.choose (num_key_projects - 1) (select_key - 1) * Nat.choose (num_general_projects - 1) (select_general - 1)) = 60 := by
  sorry


end num_selections_with_A_or_B_l267_26720


namespace margo_travel_distance_l267_26702

/-- The total distance Margo traveled given her jogging and walking times and average speed -/
theorem margo_travel_distance (jog_time walk_time avg_speed : ℝ) : 
  jog_time = 12 / 60 →
  walk_time = 25 / 60 →
  avg_speed = 5 →
  avg_speed * (jog_time + walk_time) = 3.085 :=
by sorry

end margo_travel_distance_l267_26702


namespace committee_selection_l267_26727

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) :
  Nat.choose n k = 792 := by
  sorry

end committee_selection_l267_26727


namespace rental_van_cost_increase_l267_26733

theorem rental_van_cost_increase 
  (total_cost : ℝ) 
  (initial_people : ℕ) 
  (withdrawing_people : ℕ) 
  (h1 : total_cost = 450) 
  (h2 : initial_people = 15) 
  (h3 : withdrawing_people = 3) : 
  let remaining_people := initial_people - withdrawing_people
  let initial_share := total_cost / initial_people
  let new_share := total_cost / remaining_people
  new_share - initial_share = 7.5 := by
sorry

end rental_van_cost_increase_l267_26733


namespace banana_cost_l267_26708

/-- Given that 5 dozen bananas cost $24.00, prove that 4 dozen bananas at the same rate will cost $19.20 -/
theorem banana_cost (total_cost : ℝ) (total_dozens : ℕ) (target_dozens : ℕ) 
  (h1 : total_cost = 24)
  (h2 : total_dozens = 5)
  (h3 : target_dozens = 4) :
  (target_dozens : ℝ) * (total_cost / total_dozens) = 19.2 :=
by sorry

end banana_cost_l267_26708


namespace least_common_meeting_time_l267_26791

def prime_lap_times : List Nat := [2, 3, 5, 7, 11, 13, 17]

def is_divisible_by_at_least_four (n : Nat) : Bool :=
  (prime_lap_times.filter (fun p => n % p = 0)).length ≥ 4

theorem least_common_meeting_time :
  ∃ T : Nat, T > 0 ∧ is_divisible_by_at_least_four T ∧
  ∀ t : Nat, 0 < t ∧ t < T → ¬is_divisible_by_at_least_four t :=
by sorry

end least_common_meeting_time_l267_26791


namespace prob_red_then_green_is_two_ninths_l267_26740

def num_red_balls : ℕ := 2
def num_green_balls : ℕ := 1
def total_balls : ℕ := num_red_balls + num_green_balls

def probability_red_then_green : ℚ :=
  (num_red_balls : ℚ) / total_balls * (num_green_balls : ℚ) / total_balls

theorem prob_red_then_green_is_two_ninths :
  probability_red_then_green = 2 / 9 := by
  sorry

end prob_red_then_green_is_two_ninths_l267_26740


namespace fourth_power_sum_sqrt_div_two_l267_26758

theorem fourth_power_sum_sqrt_div_two :
  Real.sqrt (4^4 + 4^4 + 4^4) / 2 = 8 * Real.sqrt 3 := by sorry

end fourth_power_sum_sqrt_div_two_l267_26758


namespace expression_equals_2x_to_4th_l267_26783

theorem expression_equals_2x_to_4th (x : ℝ) :
  let A := x^4 * x^4
  let B := x^4 + x^4
  let C := 2*x^2 + x^2
  let D := 2*x * x^4
  B = 2 * x^4 := by sorry

end expression_equals_2x_to_4th_l267_26783


namespace quadratic_sum_l267_26722

/-- The quadratic function we're working with -/
def f (x : ℝ) : ℝ := 8*x^2 + 48*x + 200

/-- The general form of a quadratic after completing the square -/
def g (a b c x : ℝ) : ℝ := a*(x+b)^2 + c

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, f x = g a b c x) → a + b + c = 139 := by
  sorry

end quadratic_sum_l267_26722


namespace problem_1_problem_2_l267_26713

-- Problem 1
theorem problem_1 : (-2023)^0 + Real.sqrt 12 + 2 * (-1/2) = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) : (2*m + 1) * (2*m - 1) - 4*m*(m - 1) = 4*m - 1 := by sorry

end problem_1_problem_2_l267_26713


namespace prob_odd_product_six_rolls_main_theorem_l267_26730

/-- A standard die has six faces numbered 1 through 6 -/
def standardDie : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- The probability of rolling an odd number on a standard die -/
def probOddRoll : Rat := 1/2

/-- The number of times the die is rolled -/
def numRolls : Nat := 6

/-- Theorem: The probability of rolling a standard die six times and obtaining an odd product is 1/64 -/
theorem prob_odd_product_six_rolls :
  (probOddRoll ^ numRolls : Rat) = 1/64 := by
  sorry

/-- Main theorem: The probability of rolling a standard die six times and obtaining an odd product is 1/64 -/
theorem main_theorem :
  ∃ (p : Rat), p = (probOddRoll ^ numRolls) ∧ p = 1/64 := by
  sorry

end prob_odd_product_six_rolls_main_theorem_l267_26730


namespace triangle_shape_l267_26724

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if (a² + b²)sin(A - B) = (a² - b²)sin(A + B),
    then the triangle is either isosceles (A = B) or right-angled (2A + 2B = 180°). -/
theorem triangle_shape (a b c A B C : ℝ) (h : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B)) :
  A = B ∨ 2*A + 2*B = Real.pi := by
  sorry

end triangle_shape_l267_26724


namespace five_digit_multiple_of_nine_l267_26797

theorem five_digit_multiple_of_nine :
  ∃ (n : ℕ), n = 56781 ∧ n % 9 = 0 := by
  sorry

end five_digit_multiple_of_nine_l267_26797


namespace sum_of_fractions_equals_seven_l267_26784

theorem sum_of_fractions_equals_seven : 
  let S := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
           1 / (Real.sqrt 12 - 3)
  S = 7 := by
  sorry

end sum_of_fractions_equals_seven_l267_26784


namespace largest_inscribed_sphere_surface_area_l267_26749

/-- The surface area of the largest sphere inscribed in a cone -/
theorem largest_inscribed_sphere_surface_area
  (base_radius : ℝ)
  (slant_height : ℝ)
  (h_base_radius : base_radius = 1)
  (h_slant_height : slant_height = 3) :
  ∃ (sphere_surface_area : ℝ),
    sphere_surface_area = 2 * Real.pi ∧
    ∀ (other_sphere_surface_area : ℝ),
      other_sphere_surface_area ≤ sphere_surface_area :=
by sorry

end largest_inscribed_sphere_surface_area_l267_26749


namespace flock_max_weight_l267_26717

/-- Represents the types of swallows --/
inductive SwallowType
| American
| European

/-- Calculates the maximum weight a swallow can carry based on its type --/
def maxWeightCarried (s : SwallowType) : ℕ :=
  match s with
  | SwallowType.American => 5
  | SwallowType.European => 10

/-- The total number of swallows in the flock --/
def totalSwallows : ℕ := 90

/-- The ratio of American to European swallows --/
def americanToEuropeanRatio : ℕ := 2

/-- Theorem stating the maximum combined weight the flock can carry --/
theorem flock_max_weight :
  let europeanCount := totalSwallows / (americanToEuropeanRatio + 1)
  let americanCount := totalSwallows - europeanCount
  europeanCount * maxWeightCarried SwallowType.European +
  americanCount * maxWeightCarried SwallowType.American = 600 := by
  sorry


end flock_max_weight_l267_26717


namespace tan_sum_problem_l267_26782

theorem tan_sum_problem (α β : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := by sorry

end tan_sum_problem_l267_26782


namespace ellipse_condition_l267_26739

/-- The equation of the graph is 9x^2 + y^2 - 36x + 8y = k -/
def graph_equation (x y k : ℝ) : Prop :=
  9 * x^2 + y^2 - 36 * x + 8 * y = k

/-- A non-degenerate ellipse has positive denominators in its standard form -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k + 52 > 0

theorem ellipse_condition (k : ℝ) :
  (∀ x y, graph_equation x y k → is_non_degenerate_ellipse k) ↔ k > -52 := by
  sorry

end ellipse_condition_l267_26739


namespace find_number_l267_26755

theorem find_number : ∃ x : ℝ, 0.62 * 150 - 0.20 * x = 43 ∧ x = 250 := by
  sorry

end find_number_l267_26755


namespace change_received_l267_26775

/-- The change received when buying gum and a protractor -/
theorem change_received (gum_cost protractor_cost amount_paid : ℕ) : 
  gum_cost = 350 → protractor_cost = 500 → amount_paid = 1000 → 
  amount_paid - (gum_cost + protractor_cost) = 150 := by
  sorry

end change_received_l267_26775


namespace fuel_tank_capacity_l267_26741

-- Define the fuel tank capacity
def C : ℝ := 200

-- Define the volume of fuel A added
def fuel_A_volume : ℝ := 349.99999999999994

-- Define the ethanol percentage in fuel A
def ethanol_A_percent : ℝ := 0.12

-- Define the ethanol percentage in fuel B
def ethanol_B_percent : ℝ := 0.16

-- Define the total ethanol volume in the full tank
def total_ethanol_volume : ℝ := 18

-- Theorem statement
theorem fuel_tank_capacity :
  C = 200 ∧
  fuel_A_volume = 349.99999999999994 ∧
  ethanol_A_percent = 0.12 ∧
  ethanol_B_percent = 0.16 ∧
  total_ethanol_volume = 18 →
  ethanol_A_percent * fuel_A_volume + ethanol_B_percent * (C - fuel_A_volume) = total_ethanol_volume :=
by sorry

end fuel_tank_capacity_l267_26741


namespace fifteen_divides_Q_largest_divisor_fifteen_largest_divisor_l267_26762

/-- The product of four consecutive positive odd integers -/
def Q (n : ℕ) : ℕ := (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3)

/-- 15 divides Q for all n -/
theorem fifteen_divides_Q (n : ℕ) : 15 ∣ Q n :=
sorry

/-- For any integer k > 15, there exists an n such that k does not divide Q n -/
theorem largest_divisor (k : ℕ) (h : k > 15) : ∃ n : ℕ, ¬(k ∣ Q n) :=
sorry

/-- 15 is the largest integer that divides Q for all n -/
theorem fifteen_largest_divisor : ∀ k : ℕ, (∀ n : ℕ, k ∣ Q n) → k ≤ 15 :=
sorry

end fifteen_divides_Q_largest_divisor_fifteen_largest_divisor_l267_26762


namespace students_not_in_biology_l267_26779

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) : 
  total_students = 880 → 
  biology_percentage = 30 / 100 →
  (total_students : ℚ) * (1 - biology_percentage) = 616 :=
by sorry

end students_not_in_biology_l267_26779


namespace polynomial_expansion_l267_26790

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 - 4 * t^2 + 5 * t - 3) * (4 * t^2 - 2 * t + 1) =
  12 * t^5 - 22 * t^4 + 31 * t^3 - 26 * t^2 + 11 * t - 3 := by
  sorry

end polynomial_expansion_l267_26790


namespace only_setA_is_pythagorean_triple_l267_26759

/-- Checks if three numbers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The sets of numbers to check -/
def setA : List ℕ := [6, 8, 10]
def setB : List ℚ := [3/10, 4/10, 5/10]
def setC : List ℚ := [3/2, 4/2, 5/2]
def setD : List ℕ := [5, 11, 12]

theorem only_setA_is_pythagorean_triple :
  (∃ (a b c : ℕ), a ∈ setA ∧ b ∈ setA ∧ c ∈ setA ∧ isPythagoreanTriple a b c) ∧
  (¬∃ (a b c : ℚ), a ∈ setB ∧ b ∈ setB ∧ c ∈ setB ∧ a.num * a.num + b.num * b.num = c.num * c.num) ∧
  (¬∃ (a b c : ℚ), a ∈ setC ∧ b ∈ setC ∧ c ∈ setC ∧ a.num * a.num + b.num * b.num = c.num * c.num) ∧
  (¬∃ (a b c : ℕ), a ∈ setD ∧ b ∈ setD ∧ c ∈ setD ∧ isPythagoreanTriple a b c) :=
by sorry


end only_setA_is_pythagorean_triple_l267_26759


namespace triangle_area_prove_triangle_area_l267_26735

/-- The area of the triangle formed by the lines y = 3x - 3, y = -2x + 18, and the y-axis -/
theorem triangle_area : ℝ → Prop :=
  fun A => 
    let line1 := fun x : ℝ => 3 * x - 3
    let line2 := fun x : ℝ => -2 * x + 18
    let y_axis := fun x : ℝ => 0
    let intersection_x := (21 : ℝ) / 5
    let intersection_y := line1 intersection_x
    let base := line2 0 - line1 0
    let height := intersection_x
    A = (1 / 2) * base * height ∧ A = 441 / 10

/-- Proof of the theorem -/
theorem prove_triangle_area : ∃ A : ℝ, triangle_area A :=
  sorry

end triangle_area_prove_triangle_area_l267_26735


namespace earth_inhabitable_fraction_l267_26726

theorem earth_inhabitable_fraction :
  let earth_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (1 : ℚ) / 3
  (land_fraction * inhabitable_land_fraction) * earth_surface = (1 : ℚ) / 9 := by
  sorry

end earth_inhabitable_fraction_l267_26726


namespace monomial_sum_exponent_l267_26756

theorem monomial_sum_exponent (m n : ℤ) : 
  (∃ k : ℤ, ∃ c : ℚ, -x^(m-2) * y^3 + 2/3 * x^n * y^(2*m-3*n) = c * x^k * y^k) → 
  m^(-n : ℤ) = (1 : ℚ)/3 :=
by sorry

end monomial_sum_exponent_l267_26756


namespace imaginary_part_of_2_minus_3i_l267_26719

theorem imaginary_part_of_2_minus_3i :
  Complex.im (2 - 3 * Complex.I) = -3 := by
  sorry

end imaginary_part_of_2_minus_3i_l267_26719


namespace a_gt_one_sufficient_not_necessary_l267_26704

theorem a_gt_one_sufficient_not_necessary (a : ℝ) (h : a ≠ 0) :
  (∀ a, a > 1 → a > 1/a) ∧ (∃ a, a > 1/a ∧ a ≤ 1) := by sorry

end a_gt_one_sufficient_not_necessary_l267_26704


namespace students_left_l267_26716

theorem students_left (initial_boys initial_girls boys_dropout girls_dropout : ℕ) 
  (h1 : initial_boys = 14)
  (h2 : initial_girls = 10)
  (h3 : boys_dropout = 4)
  (h4 : girls_dropout = 3) :
  initial_boys - boys_dropout + (initial_girls - girls_dropout) = 17 := by
  sorry

end students_left_l267_26716


namespace money_saved_calculation_marcus_shopping_savings_l267_26737

/-- Calculates the money saved when buying discounted items with sales tax --/
theorem money_saved_calculation (max_budget : ℝ) 
  (shoe_price shoe_discount : ℝ) 
  (sock_price sock_discount : ℝ) 
  (shirt_price shirt_discount : ℝ) 
  (sales_tax : ℝ) : ℝ :=
  let discounted_shoe := shoe_price * (1 - shoe_discount)
  let discounted_sock := sock_price * (1 - sock_discount)
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let total_before_tax := discounted_shoe + discounted_sock + discounted_shirt
  let final_cost := total_before_tax * (1 + sales_tax)
  let money_saved := max_budget - final_cost
  money_saved

/-- Proves that the money saved is approximately $34.22 --/
theorem marcus_shopping_savings : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |money_saved_calculation 200 120 0.3 25 0.2 55 0.1 0.08 - 34.22| < ε := by
  sorry

end money_saved_calculation_marcus_shopping_savings_l267_26737


namespace third_term_not_unique_l267_26753

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The product of the first 5 terms of a sequence equals 32 -/
def ProductEquals32 (a : ℕ → ℝ) : Prop :=
  a 1 * a 2 * a 3 * a 4 * a 5 = 32

/-- The third term of a geometric sequence cannot be uniquely determined
    given only that the product of the first 5 terms equals 32 -/
theorem third_term_not_unique (a : ℕ → ℝ) 
    (h1 : GeometricSequence a) (h2 : ProductEquals32 a) :
    ¬∃! x : ℝ, a 3 = x :=
  sorry

end third_term_not_unique_l267_26753


namespace not_divisible_by_121_l267_26750

theorem not_divisible_by_121 (n : ℤ) : ¬(∃ (k : ℤ), n^2 + 3*n + 5 = 121*k ∨ n^2 - 3*n + 5 = 121*k) := by
  sorry

end not_divisible_by_121_l267_26750


namespace imaginary_part_of_z_l267_26744

theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / Complex.I = 1 - z) : 
  Complex.im z = 1 := by
  sorry

end imaginary_part_of_z_l267_26744


namespace building_painting_cost_l267_26732

theorem building_painting_cost (room1_area room2_area room3_area : ℝ)
  (paint_price1 paint_price2 paint_price3 : ℝ)
  (labor_cost : ℝ) (tax_rate : ℝ) :
  room1_area = 196 →
  room2_area = 150 →
  room3_area = 250 →
  paint_price1 = 15 →
  paint_price2 = 18 →
  paint_price3 = 20 →
  labor_cost = 800 →
  tax_rate = 0.05 →
  let room1_cost := room1_area * paint_price1
  let room2_cost := room2_area * paint_price2
  let room3_cost := room3_area * paint_price3
  let total_painting_cost := room1_cost + room2_cost + room3_cost
  let total_cost_before_tax := total_painting_cost + labor_cost
  let tax := total_cost_before_tax * tax_rate
  let total_cost_after_tax := total_cost_before_tax + tax
  total_cost_after_tax = 12012 :=
by sorry

end building_painting_cost_l267_26732


namespace ball_distribution_l267_26781

/-- Represents the number of ways to distribute balls into boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to place 7 distinguishable balls into 3 boxes,
    where one box is red and the other two are indistinguishable -/
theorem ball_distribution : distribute_balls 7 3 = 64 := by sorry

end ball_distribution_l267_26781


namespace single_digit_integer_equation_l267_26799

theorem single_digit_integer_equation : ∃ (x a y z b : ℕ),
  (0 < x ∧ x < 10) ∧
  (0 < a ∧ a < 10) ∧
  (0 < y ∧ y < 10) ∧
  (0 < z ∧ z < 10) ∧
  (0 < b ∧ b < 10) ∧
  (x = a / 6) ∧
  (z = b / 6) ∧
  (y = (a + b) % 5) ∧
  (100 * x + 10 * y + z = 121) :=
by
  sorry

end single_digit_integer_equation_l267_26799


namespace factorization_equality_l267_26746

theorem factorization_equality (a b : ℝ) :
  276 * a^2 * b^2 + 69 * a * b - 138 * a * b^3 = 69 * a * b * (4 * a * b + 1 - 2 * b^2) := by
  sorry

end factorization_equality_l267_26746


namespace green_then_blue_probability_l267_26742

/-- The probability of drawing a green marble first and a blue marble second from a bag -/
theorem green_then_blue_probability 
  (total_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (green_marbles : ℕ) 
  (h1 : total_marbles = blue_marbles + green_marbles)
  (h2 : blue_marbles = 4)
  (h3 : green_marbles = 6) :
  (green_marbles : ℚ) / total_marbles * (blue_marbles : ℚ) / (total_marbles - 1) = 4 / 15 := by
  sorry

end green_then_blue_probability_l267_26742


namespace imaginary_power_sum_l267_26715

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^11 + i^111 = -2 * i := by sorry

end imaginary_power_sum_l267_26715


namespace round_31083_58_to_two_sig_figs_l267_26709

/-- Rounds a number to a specified number of significant figures -/
def roundToSignificantFigures (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Theorem: Rounding 31,083.58 to two significant figures results in 3.1 × 10^4 -/
theorem round_31083_58_to_two_sig_figs :
  roundToSignificantFigures 31083.58 2 = 3.1 * 10^4 := by sorry

end round_31083_58_to_two_sig_figs_l267_26709


namespace valid_assignments_l267_26770

-- Define a type for statements
inductive Statement
| Assign1 : Statement  -- x←1, y←2, z←3
| Assign2 : Statement  -- S^2←4
| Assign3 : Statement  -- i←i+2
| Assign4 : Statement  -- x+1←x

-- Define a predicate for valid assignment statements
def is_valid_assignment (s : Statement) : Prop :=
  match s with
  | Statement.Assign1 => True
  | Statement.Assign2 => False
  | Statement.Assign3 => True
  | Statement.Assign4 => False

-- Theorem stating which statements are valid assignments
theorem valid_assignments :
  (is_valid_assignment Statement.Assign1) ∧
  (¬is_valid_assignment Statement.Assign2) ∧
  (is_valid_assignment Statement.Assign3) ∧
  (¬is_valid_assignment Statement.Assign4) := by
  sorry

end valid_assignments_l267_26770


namespace interest_rate_calculation_l267_26729

/-- Given a principal amount and an interest rate, proves that if the simple interest
    for 2 years is 40 and the compound interest for 2 years is 41, then the interest rate is 5% -/
theorem interest_rate_calculation (P : ℝ) (r : ℝ) 
    (h1 : P * r * 2 = 40)  -- Simple interest condition
    (h2 : P * ((1 + r)^2 - 1) = 41) -- Compound interest condition
    : r = 0.05 := by
  sorry

#check interest_rate_calculation

end interest_rate_calculation_l267_26729


namespace sphere_center_sum_l267_26776

theorem sphere_center_sum (x y z : ℝ) :
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 → x + y + z = 2 := by
  sorry

end sphere_center_sum_l267_26776


namespace side_median_ratio_not_unique_l267_26757

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of a side to its corresponding median in a triangle --/
def sideMedianRatio (t : Triangle) : ℝ := 
  sorry

/-- Predicate to check if two triangles have the same shape (are similar) --/
def hasSameShape (t1 t2 : Triangle) : Prop := 
  sorry

/-- Theorem stating that the ratio of a side to its corresponding median 
    does not uniquely determine a triangle's shape --/
theorem side_median_ratio_not_unique : 
  ∃ t1 t2 : Triangle, 
    sideMedianRatio t1 = sideMedianRatio t2 ∧ 
    ¬(hasSameShape t1 t2) := by
  sorry

end side_median_ratio_not_unique_l267_26757


namespace apples_taken_per_basket_l267_26765

theorem apples_taken_per_basket (initial_apples : ℕ) (num_baskets : ℕ) (apples_per_basket : ℕ) :
  initial_apples = 64 →
  num_baskets = 4 →
  apples_per_basket = 13 →
  ∃ (taken_per_basket : ℕ),
    taken_per_basket * num_baskets = initial_apples - (apples_per_basket * num_baskets) ∧
    taken_per_basket = 3 :=
by sorry

end apples_taken_per_basket_l267_26765


namespace money_ratio_proof_l267_26703

theorem money_ratio_proof (natasha_money carla_money cosima_money : ℚ) :
  natasha_money = 3 * carla_money →
  carla_money = cosima_money →
  natasha_money = 60 →
  (7 / 5) * (natasha_money + carla_money + cosima_money) - (natasha_money + carla_money + cosima_money) = 36 →
  carla_money / cosima_money = 1 := by
  sorry

end money_ratio_proof_l267_26703


namespace triangle_angle_c_value_l267_26736

theorem triangle_angle_c_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a^2 = 3*b^2 + 3*c^2 - 2*Real.sqrt 3*b*c*Real.sin A →
  C = π/6 := by
sorry

end triangle_angle_c_value_l267_26736


namespace intersection_ratio_l267_26786

-- Define the slopes and y-intercepts of the two lines
variable (k₁ k₂ : ℝ)

-- Define the condition that the lines intersect on the x-axis
def intersect_on_x_axis (k₁ k₂ : ℝ) : Prop :=
  ∃ x : ℝ, k₁ * x + 4 = 0 ∧ k₂ * x - 2 = 0

-- Theorem statement
theorem intersection_ratio (k₁ k₂ : ℝ) (h : intersect_on_x_axis k₁ k₂) (h₁ : k₁ ≠ 0) (h₂ : k₂ ≠ 0) :
  k₁ / k₂ = -2 :=
sorry

end intersection_ratio_l267_26786


namespace custom_op_three_six_l267_26769

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a.val ^ 2 * b.val : ℚ) / (a.val + b.val)

/-- Theorem stating that 3 @ 6 = 6 -/
theorem custom_op_three_six :
  custom_op 3 6 = 6 := by sorry

end custom_op_three_six_l267_26769


namespace exists_diverse_line_l267_26721

/-- Represents a 17x17 table with integers from 1 to 17 -/
def Table := Fin 17 → Fin 17 → Fin 17

/-- Predicate to check if a table is valid according to the problem conditions -/
def is_valid_table (t : Table) : Prop :=
  ∀ n : Fin 17, (Finset.univ.filter (λ (i : Fin 17 × Fin 17) => t i.1 i.2 = n)).card = 17

/-- Counts the number of different elements in a list -/
def count_different (l : List (Fin 17)) : Nat :=
  (l.toFinset).card

/-- Theorem stating the existence of a row or column with at least 5 different numbers -/
theorem exists_diverse_line (t : Table) (h : is_valid_table t) :
  (∃ i : Fin 17, count_different (List.ofFn (λ j => t i j)) ≥ 5) ∨
  (∃ j : Fin 17, count_different (List.ofFn (λ i => t i j)) ≥ 5) := by
  sorry

end exists_diverse_line_l267_26721


namespace quadratic_inequality_range_l267_26743

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, (a + 1) * x^2 + 4 * x + 1 < 0) ↔ a < 1 :=
by sorry

end quadratic_inequality_range_l267_26743


namespace theresa_crayons_l267_26795

/-- Theresa's initial number of crayons -/
def theresa_initial : ℕ := sorry

/-- Theresa's number of crayons after sharing -/
def theresa_after : ℕ := 19

/-- Janice's initial number of crayons -/
def janice_initial : ℕ := 12

/-- Number of crayons Janice shares with Nancy -/
def janice_shares : ℕ := 13

theorem theresa_crayons : theresa_initial = theresa_after := by sorry

end theresa_crayons_l267_26795


namespace cubic_roots_sum_l267_26763

theorem cubic_roots_sum (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 8*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  (k + m = 27 ∨ k + m = 31) :=
sorry

end cubic_roots_sum_l267_26763


namespace A_eq_set_zero_one_two_l267_26764

def A : Set ℤ := {x | -1 < |x - 1| ∧ |x - 1| < 2}

theorem A_eq_set_zero_one_two : A = {0, 1, 2} := by sorry

end A_eq_set_zero_one_two_l267_26764


namespace harris_flour_amount_l267_26767

theorem harris_flour_amount (flour_per_cake : ℕ) (total_cakes : ℕ) (traci_flour : ℕ) :
  flour_per_cake = 100 →
  total_cakes = 9 →
  traci_flour = 500 →
  flour_per_cake * total_cakes - traci_flour = 400 := by
sorry

end harris_flour_amount_l267_26767


namespace total_jog_time_two_weeks_l267_26761

/-- The number of hours jogged daily -/
def daily_jog_hours : ℝ := 1.5

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- Theorem: Total jogging time in two weeks -/
theorem total_jog_time_two_weeks : 
  daily_jog_hours * (days_in_two_weeks : ℝ) = 21 := by
  sorry

end total_jog_time_two_weeks_l267_26761


namespace nine_powers_equal_three_power_l267_26711

theorem nine_powers_equal_three_power (n : ℕ) : 
  9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n = 3^2012 → n = 1005 := by
  sorry

end nine_powers_equal_three_power_l267_26711


namespace refrigerator_transport_cost_prove_transport_cost_l267_26772

/-- Calculates the transport cost for a refrigerator purchase --/
theorem refrigerator_transport_cost 
  (purchase_price : ℝ) 
  (discount_rate : ℝ) 
  (installation_cost : ℝ) 
  (profit_rate : ℝ) 
  (selling_price : ℝ) : ℝ :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let total_cost := purchase_price + installation_cost
  let transport_cost := (selling_price / (1 + profit_rate)) - total_cost
  transport_cost

/-- Proves that the transport cost is 4000 given the problem conditions --/
theorem prove_transport_cost : 
  refrigerator_transport_cost 15500 0.2 250 0.1 21725 = 4000 := by
  sorry

end refrigerator_transport_cost_prove_transport_cost_l267_26772


namespace inequality_existence_l267_26725

variable (a : ℝ)

theorem inequality_existence (h1 : a > 1) (h2 : a ≠ 2) :
  (¬ ∀ x : ℝ, (1 < x ∧ x < a) → (a < 2*x ∧ 2*x < a^2)) ∧
  (∃ x : ℝ, (a < 2*x ∧ 2*x < a^2) ∧ ¬(1 < x ∧ x < a)) := by
  sorry

end inequality_existence_l267_26725


namespace arithmetic_mean_of_fractions_l267_26700

theorem arithmetic_mean_of_fractions :
  let f1 : ℚ := 3 / 8
  let f2 : ℚ := 5 / 9
  let f3 : ℚ := 7 / 12
  let mean : ℚ := (f1 + f2 + f3) / 3
  mean = 109 / 216 := by
  sorry

end arithmetic_mean_of_fractions_l267_26700


namespace average_salary_l267_26760

/-- The average salary of 5 people with given salaries is 9000 --/
theorem average_salary (a b c d e : ℕ) 
  (ha : a = 8000) (hb : b = 5000) (hc : c = 16000) (hd : d = 7000) (he : e = 9000) : 
  (a + b + c + d + e) / 5 = 9000 := by
  sorry

end average_salary_l267_26760


namespace count_polygons_l267_26731

/-- The number of points placed on the circle -/
def n : ℕ := 15

/-- The number of distinct convex polygons with at least three sides -/
def num_polygons : ℕ := 2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2)

/-- Theorem stating that the number of distinct convex polygons is 32647 -/
theorem count_polygons : num_polygons = 32647 := by
  sorry

end count_polygons_l267_26731


namespace nonagon_diagonals_l267_26745

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  is_convex : Bool
  right_angles : ℕ

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex 9-sided polygon with three right angles has 27 diagonals -/
theorem nonagon_diagonals (P : ConvexPolygon 9) (h1 : P.is_convex = true) (h2 : P.right_angles = 3) :
  num_diagonals P.sides = 27 := by
  sorry

end nonagon_diagonals_l267_26745


namespace equilateral_triangle_area_perimeter_ratio_l267_26728

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / (perimeter ^ 2) = Real.sqrt 3 / 36 := by
sorry

end equilateral_triangle_area_perimeter_ratio_l267_26728


namespace sector_area_l267_26714

/-- Given a circular sector with central angle 120° and arc length 6π, its area is 27π -/
theorem sector_area (θ : ℝ) (arc_length : ℝ) (area : ℝ) : 
  θ = 120 * π / 180 →  -- Convert 120° to radians
  arc_length = 6 * π → 
  area = 27 * π :=
by
  sorry


end sector_area_l267_26714


namespace stock_price_after_two_years_l267_26718

def initial_price : ℝ := 200
def first_year_increase : ℝ := 0.50
def second_year_decrease : ℝ := 0.30

theorem stock_price_after_two_years :
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price = 210 := by sorry

end stock_price_after_two_years_l267_26718


namespace solve_equation_l267_26701

theorem solve_equation (x : ℝ) (h : (128 / x) + (75 / x) + (57 / x) = 6.5) : x = 40 := by
  sorry

end solve_equation_l267_26701


namespace fraction_equation_solution_l267_26712

theorem fraction_equation_solution (x : ℝ) : 
  (3 - x) / (2 - x) - 1 / (x - 2) = 3 → x = 1 := by
  sorry

end fraction_equation_solution_l267_26712


namespace fixed_point_of_exponential_function_l267_26794

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x : ℝ => a^(x + 1) - 1
  f (-1) = 0 := by
sorry

end fixed_point_of_exponential_function_l267_26794


namespace arithmetic_sequence_properties_l267_26723

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ  -- The sequence
  d : ℝ        -- Common difference
  S : ℕ+ → ℝ  -- Sum function
  sum_def : ∀ n : ℕ+, S n = n * a 1 + n * (n - 1) / 2 * d
  seq_def : ∀ n : ℕ+, a n = a 1 + (n - 1) * d

/-- Theorem about properties of an arithmetic sequence given certain conditions -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h : seq.S 6 > seq.S 7 ∧ seq.S 7 > seq.S 5) :
    seq.d < 0 ∧ 
    seq.S 11 > 0 ∧ 
    seq.S 12 > 0 ∧ 
    seq.S 13 < 0 := by
  sorry

end arithmetic_sequence_properties_l267_26723


namespace consecutive_product_not_power_l267_26734

theorem consecutive_product_not_power (n m : ℕ) (h : m > 1) :
  ¬ ∃ k : ℕ, (n - 1) * n * (n + 1) = k ^ m := by
  sorry

end consecutive_product_not_power_l267_26734


namespace min_value_fraction_l267_26778

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 1/a + 2/b ≤ 1/x + 2/y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 1/x + 2/y = (3 + 2 * Real.sqrt 2) / 2) := by
  sorry

end min_value_fraction_l267_26778


namespace weight_difference_l267_26792

theorem weight_difference (n : ℕ) (joe_weight : ℝ) (initial_avg : ℝ) (new_avg : ℝ) :
  joe_weight = 42 →
  initial_avg = 30 →
  new_avg = 31 →
  (n * initial_avg + joe_weight) / (n + 1) = new_avg →
  let total_weight := n * initial_avg + joe_weight
  let remaining_students := n - 1
  ∃ (x : ℝ), (total_weight - 2 * x) / remaining_students = initial_avg →
  |x - joe_weight| = 6 :=
by sorry

end weight_difference_l267_26792


namespace ellipse_major_axis_length_l267_26774

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 9 = 1

/-- Definition of the major axis length -/
def major_axis_length (f : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem: The length of the major axis of the ellipse is 6 -/
theorem ellipse_major_axis_length :
  major_axis_length is_ellipse = 6 := by sorry

end ellipse_major_axis_length_l267_26774


namespace prime_divides_n6_minus_1_implies_n_greater_than_sqrt_p_minus_1_l267_26773

theorem prime_divides_n6_minus_1_implies_n_greater_than_sqrt_p_minus_1 
  (p : ℕ) (n : ℕ) (h_prime : Nat.Prime p) (h_n_ge_2 : n ≥ 2) 
  (h_div : p ∣ (n^6 - 1)) : n > Real.sqrt p - 1 :=
sorry

end prime_divides_n6_minus_1_implies_n_greater_than_sqrt_p_minus_1_l267_26773


namespace solution_to_system_l267_26705

theorem solution_to_system (x y z : ℝ) 
  (eq1 : x = 1 + Real.sqrt (y - z^2))
  (eq2 : y = 1 + Real.sqrt (z - x^2))
  (eq3 : z = 1 + Real.sqrt (x - y^2)) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end solution_to_system_l267_26705


namespace continuous_function_characterization_l267_26754

theorem continuous_function_characterization
  (f : ℝ → ℝ)
  (hf_continuous : Continuous f)
  (hf_zero : f 0 = 0)
  (hf_ineq : ∀ x y : ℝ, f (x^2 - y^2) ≥ x * f x - y * f y) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end continuous_function_characterization_l267_26754


namespace ab_squared_commutes_l267_26789

theorem ab_squared_commutes (a b : ℝ) : a * b^2 - b^2 * a = 0 := by
  sorry

end ab_squared_commutes_l267_26789


namespace equal_intercept_line_properties_l267_26710

/-- A line passing through (1, 2) with equal intercepts on both axes -/
def equal_intercept_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 3}

theorem equal_intercept_line_properties :
  (1, 2) ∈ equal_intercept_line ∧
  ∃ a : ℝ, a ≠ 0 ∧ (a, 0) ∈ equal_intercept_line ∧ (0, a) ∈ equal_intercept_line :=
by sorry

end equal_intercept_line_properties_l267_26710


namespace circle_center_problem_circle_center_l267_26777

/-- The equation of a circle in the form x² + y² + 2ax + 2by + c = 0 
    has center (-a, -b) -/
theorem circle_center (a b c : ℝ) :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 + 2*a*x + 2*b*y + c = 0
  let center := (-a, -b)
  ∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = a^2 + b^2 - c :=
by sorry

/-- The center of the circle x² + y² + 2x + 4y - 3 = 0 is (-1, -2) -/
theorem problem_circle_center :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 + 2*x + 4*y - 3 = 0
  let center := (-1, -2)
  ∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = 8 :=
by sorry

end circle_center_problem_circle_center_l267_26777


namespace fourth_root_equation_implies_x_power_eight_zero_l267_26788

theorem fourth_root_equation_implies_x_power_eight_zero (x : ℝ) :
  (((1 - x^4 : ℝ)^(1/4) + (1 + x^4 : ℝ)^(1/4)) = 2) → x^8 = 0 := by
  sorry

end fourth_root_equation_implies_x_power_eight_zero_l267_26788
