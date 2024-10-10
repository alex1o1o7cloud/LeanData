import Mathlib

namespace teachers_survey_l212_21230

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ) :
  total = 150 →
  high_bp = 90 →
  heart_trouble = 50 →
  both = 30 →
  (((total - (high_bp + heart_trouble - both)) : ℚ) / total) * 100 = 80 / 3 :=
by sorry

end teachers_survey_l212_21230


namespace polynomial_roots_l212_21261

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 13*x - 15

-- State the theorem
theorem polynomial_roots :
  (∃ a b c : ℝ, a < 0 ∧ 0 < b ∧ 0 < c ∧
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c)) :=
by sorry

end polynomial_roots_l212_21261


namespace sin_ratio_comparison_l212_21288

theorem sin_ratio_comparison : (Real.sin (3 * π / 180)) / (Real.sin (4 * π / 180)) > (Real.sin (1 * π / 180)) / (Real.sin (2 * π / 180)) := by
  sorry

end sin_ratio_comparison_l212_21288


namespace factorial_division_l212_21244

theorem factorial_division : 
  (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 151200 := by
  sorry

end factorial_division_l212_21244


namespace system_solution_l212_21274

theorem system_solution : 
  ∃ (x y : ℚ), 2 * x + 3 * y = 1 ∧ 3 * x - 6 * y = 7 ∧ x = 9/7 ∧ y = -11/21 := by
  sorry

end system_solution_l212_21274


namespace solve_system_l212_21259

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -25 / 11 := by sorry

end solve_system_l212_21259


namespace two_digit_average_decimal_l212_21219

theorem two_digit_average_decimal (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 →  -- m and n are 2-digit positive integers
  (m + n) / 2 = m + n / 100 →            -- their average equals the decimal m.n
  min m n = 49 :=                        -- the smaller of m and n is 49
by sorry

end two_digit_average_decimal_l212_21219


namespace sector_central_angle_l212_21237

theorem sector_central_angle (s : Real) (r : Real) (θ : Real) 
  (h1 : s = π) 
  (h2 : r = 2) 
  (h3 : s = r * θ) : θ = π / 2 := by
  sorry

end sector_central_angle_l212_21237


namespace roots_of_quadratic_l212_21226

theorem roots_of_quadratic (x : ℝ) : x^2 = 5*x ↔ x = 0 ∨ x = 5 := by
  sorry

end roots_of_quadratic_l212_21226


namespace problem_4_l212_21249

theorem problem_4 (a : ℝ) : (2*a + 1)^2 - (2*a + 1)*(2*a - 1) = 4*a + 2 := by
  sorry

end problem_4_l212_21249


namespace x_value_l212_21266

theorem x_value : ∃ x : ℝ, 3 * x = (26 - x) + 18 ∧ x = 11 := by sorry

end x_value_l212_21266


namespace sam_yellow_marbles_l212_21236

/-- The number of yellow marbles Sam has after receiving more from Joan -/
def total_yellow_marbles (initial : Real) (received : Real) : Real :=
  initial + received

/-- Theorem stating that Sam now has 111.0 yellow marbles -/
theorem sam_yellow_marbles :
  total_yellow_marbles 86.0 25.0 = 111.0 := by
  sorry

end sam_yellow_marbles_l212_21236


namespace max_volume_rectangular_prism_l212_21287

/-- Represents the volume of a rectangular prism as a function of the shorter base edge length -/
def prism_volume (x : ℝ) : ℝ := x * (x + 0.5) * (3.2 - 2 * x)

/-- The theorem stating the maximum volume and corresponding height of the rectangular prism -/
theorem max_volume_rectangular_prism :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < 1.6 ∧
    prism_volume x = 1.8 ∧
    3.2 - 2 * x = 1.2 ∧
    ∀ (y : ℝ), 0 < y ∧ y < 1.6 → prism_volume y ≤ prism_volume x :=
sorry


end max_volume_rectangular_prism_l212_21287


namespace x_value_l212_21207

/-- Given that ( √x ) / ( √0.81 ) + ( √1.44 ) / ( √0.49 ) = 2.879628878919216, prove that x = 1.1 -/
theorem x_value (x : ℝ) 
  (h : Real.sqrt x / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt 0.49 = 2.879628878919216) : 
  x = 1.1 := by
  sorry

end x_value_l212_21207


namespace range_of_sum_l212_21293

theorem range_of_sum (a b : ℝ) :
  (∀ x : ℝ, |x - a| + |x + b| ≥ 3) →
  a + b ∈ Set.Iic (-3) ∪ Set.Ioi 3 :=
by
  sorry

end range_of_sum_l212_21293


namespace sqrt_inequality_l212_21282

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt a - Real.sqrt (a - 2) < Real.sqrt (a - 1) - Real.sqrt (a - 3) := by
  sorry

end sqrt_inequality_l212_21282


namespace percentage_problem_l212_21265

theorem percentage_problem : ∃ p : ℝ, p > 0 ∧ p < 100 ∧ (p / 100) * 30 = (25 / 100) * 16 + 2 := by
  sorry

end percentage_problem_l212_21265


namespace assignment_effect_l212_21201

/-- Represents the effect of the assignment statement M = M + 3 --/
theorem assignment_effect (M : ℤ) : 
  let M' := M + 3
  M' = M + 3 := by sorry

end assignment_effect_l212_21201


namespace irreducible_fractions_l212_21248

theorem irreducible_fractions (a b m n : ℕ) (h_n : n > 0) :
  (Nat.gcd a b = 1 → Nat.gcd (b - a) b = 1) ∧
  (Nat.gcd (m - n) (m + n) = 1 → Nat.gcd m n = 1) ∧
  (∃ (k : ℕ), (5 * n + 2) = k * (10 * n + 7) → Nat.gcd (5 * n + 2) (10 * n + 7) = 3) :=
by sorry

end irreducible_fractions_l212_21248


namespace rabbit_weight_l212_21228

/-- Given the weights of a rabbit and two guinea pigs satisfying certain conditions,
    prove that the rabbit weighs 5 pounds. -/
theorem rabbit_weight (a b c : ℝ) 
  (total_weight : a + b + c = 30)
  (larger_smaller : a + c = 2 * b)
  (rabbit_smaller : a + b = c) : 
  a = 5 := by sorry

end rabbit_weight_l212_21228


namespace opposite_of_negative_fraction_l212_21251

theorem opposite_of_negative_fraction :
  -(-(1 : ℚ) / 2023) = 1 / 2023 := by sorry

end opposite_of_negative_fraction_l212_21251


namespace log_inequality_l212_21208

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by
  sorry

end log_inequality_l212_21208


namespace added_number_after_doubling_l212_21210

theorem added_number_after_doubling (initial_number : ℕ) (added_number : ℕ) : 
  initial_number = 9 →
  3 * (2 * initial_number + added_number) = 93 →
  added_number = 13 := by
sorry

end added_number_after_doubling_l212_21210


namespace video_game_lives_l212_21256

theorem video_game_lives (initial lives_lost lives_gained : ℕ) :
  initial ≥ lives_lost →
  initial - lives_lost + lives_gained = initial + lives_gained - lives_lost :=
by sorry

end video_game_lives_l212_21256


namespace largest_common_term_l212_21231

def first_sequence (n : ℕ) : ℕ := 3 + 10 * (n - 1)
def second_sequence (n : ℕ) : ℕ := 5 + 8 * (n - 1)

theorem largest_common_term : 
  (∃ (n m : ℕ), first_sequence n = second_sequence m ∧ first_sequence n = 133) ∧
  (∀ (x : ℕ), x > 133 → x ≤ 150 → 
    (∀ (n m : ℕ), first_sequence n ≠ x ∨ second_sequence m ≠ x)) :=
by sorry

end largest_common_term_l212_21231


namespace intersection_point_l212_21202

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y + 2 = 0

def C₂ (x y : ℝ) : Prop := y^2 = 8*x

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ, C₁ p.1 p.2 ∧ C₂ p.1 p.2 ∧ p = (2, -4) := by
  sorry

end intersection_point_l212_21202


namespace g_symmetric_about_one_l212_21296

-- Define the real-valued functions f and g
variable (f : ℝ → ℝ)
def g (x : ℝ) : ℝ := f (|x - 1|)

-- Define symmetry about a vertical line
def symmetric_about_line (h : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, h (a + x) = h (a - x)

-- Theorem statement
theorem g_symmetric_about_one (f : ℝ → ℝ) :
  symmetric_about_line (g f) 1 := by
  sorry

end g_symmetric_about_one_l212_21296


namespace sin_330_degrees_l212_21232

theorem sin_330_degrees : Real.sin (330 * Real.pi / 180) = -1/2 := by
  sorry

end sin_330_degrees_l212_21232


namespace cos_a_minus_pi_l212_21298

theorem cos_a_minus_pi (a : Real) 
  (h1 : π / 2 < a ∧ a < π) 
  (h2 : 3 * Real.sin (2 * a) = 2 * Real.cos a) : 
  Real.cos (a - π) = 2 * Real.sqrt 2 / 3 := by
  sorry

end cos_a_minus_pi_l212_21298


namespace average_score_is_490_l212_21281

-- Define the maximum score
def max_score : ℕ := 700

-- Define the number of students
def num_students : ℕ := 4

-- Define the scores as percentages
def gibi_percent : ℕ := 59
def jigi_percent : ℕ := 55
def mike_percent : ℕ := 99
def lizzy_percent : ℕ := 67

-- Define a function to calculate the actual score from a percentage
def calculate_score (percent : ℕ) : ℕ :=
  (percent * max_score) / 100

-- Theorem to prove
theorem average_score_is_490 : 
  (calculate_score gibi_percent + calculate_score jigi_percent + 
   calculate_score mike_percent + calculate_score lizzy_percent) / num_students = 490 :=
by sorry

end average_score_is_490_l212_21281


namespace napkin_ratio_l212_21268

/-- Proves the ratio of napkins Amelia gave to napkins Olivia gave -/
theorem napkin_ratio (william_initial : ℕ) (william_final : ℕ) (olivia_gave : ℕ) 
  (h1 : william_initial = 15)
  (h2 : william_final = 45)
  (h3 : olivia_gave = 10) :
  (william_final - william_initial - olivia_gave) / olivia_gave = 2 := by
  sorry

end napkin_ratio_l212_21268


namespace helen_lawn_mowing_gas_usage_l212_21206

/-- Represents the lawn cutting schedule and gas usage for Helen's lawn mowing --/
structure LawnCuttingSchedule where
  march_to_october_low_freq : Nat  -- Number of months with 2 cuts per month
  may_to_august_high_freq : Nat    -- Number of months with 4 cuts per month
  cuts_per_low_freq_month : Nat    -- Number of cuts in low frequency months
  cuts_per_high_freq_month : Nat   -- Number of cuts in high frequency months
  gas_usage_frequency : Nat        -- Every nth cut uses gas
  gas_usage_amount : Nat           -- Amount of gas used every nth cut

/-- Calculates the total gas usage for Helen's lawn mowing schedule --/
def calculate_gas_usage (schedule : LawnCuttingSchedule) : Nat :=
  let total_cuts := 
    schedule.march_to_october_low_freq * schedule.cuts_per_low_freq_month +
    schedule.may_to_august_high_freq * schedule.cuts_per_high_freq_month
  let gas_usage_times := total_cuts / schedule.gas_usage_frequency
  gas_usage_times * schedule.gas_usage_amount

/-- Theorem stating that Helen's lawn mowing schedule results in 12 gallons of gas usage --/
theorem helen_lawn_mowing_gas_usage :
  let schedule : LawnCuttingSchedule := {
    march_to_october_low_freq := 4
    may_to_august_high_freq := 4
    cuts_per_low_freq_month := 2
    cuts_per_high_freq_month := 4
    gas_usage_frequency := 4
    gas_usage_amount := 2
  }
  calculate_gas_usage schedule = 12 := by
  sorry


end helen_lawn_mowing_gas_usage_l212_21206


namespace positive_X_value_l212_21227

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem positive_X_value (X : ℝ) :
  (hash X 7 = 170) → (X = 11 ∨ X = -11) :=
by sorry

end positive_X_value_l212_21227


namespace max_intersections_four_circles_l212_21270

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of intersections between a line and a circle --/
def intersectionCount (l : Line) (c : Circle) : ℕ := sorry

/-- Checks if four circles are coplanar --/
def areCoplanar (c1 c2 c3 c4 : Circle) : Prop := sorry

/-- Theorem: The maximum number of intersections between a line and four coplanar circles is 8 --/
theorem max_intersections_four_circles (c1 c2 c3 c4 : Circle) (l : Line) :
  areCoplanar c1 c2 c3 c4 →
  (intersectionCount l c1 + intersectionCount l c2 + intersectionCount l c3 + intersectionCount l c4) ≤ 8 :=
sorry

end max_intersections_four_circles_l212_21270


namespace total_soap_cost_two_years_l212_21277

/-- Represents the types of soap -/
inductive SoapType
  | Lavender
  | Lemon
  | Sandalwood

/-- Represents the price of each soap type -/
def soapPrice (t : SoapType) : ℚ :=
  match t with
  | SoapType.Lavender => 4
  | SoapType.Lemon => 5
  | SoapType.Sandalwood => 6

/-- Represents the bulk discount for each soap type and quantity -/
def bulkDiscount (t : SoapType) (quantity : ℕ) : ℚ :=
  match t with
  | SoapType.Lavender =>
    if quantity ≥ 10 then 0.2
    else if quantity ≥ 5 then 0.1
    else 0
  | SoapType.Lemon =>
    if quantity ≥ 8 then 0.15
    else if quantity ≥ 4 then 0.05
    else 0
  | SoapType.Sandalwood =>
    if quantity ≥ 9 then 0.2
    else if quantity ≥ 6 then 0.1
    else if quantity ≥ 3 then 0.05
    else 0

/-- Calculates the cost of soap for a given type and quantity with bulk discount -/
def soapCost (t : SoapType) (quantity : ℕ) : ℚ :=
  let price := soapPrice t
  let discount := bulkDiscount t quantity
  quantity * price * (1 - discount)

/-- Theorem: The total amount Elias spends on soap in two years is $112.4 -/
theorem total_soap_cost_two_years :
  soapCost SoapType.Lavender 5 + soapCost SoapType.Lavender 3 +
  soapCost SoapType.Lemon 4 + soapCost SoapType.Lemon 4 +
  soapCost SoapType.Sandalwood 6 + soapCost SoapType.Sandalwood 2 = 112.4 := by
  sorry


end total_soap_cost_two_years_l212_21277


namespace other_number_l212_21278

theorem other_number (x : ℝ) : x + 0.525 = 0.650 → x = 0.125 := by
  sorry

end other_number_l212_21278


namespace total_amount_paid_l212_21269

def grape_quantity : ℕ := 8
def grape_price : ℚ := 70
def mango_quantity : ℕ := 8
def mango_price : ℚ := 55
def orange_quantity : ℕ := 5
def orange_price : ℚ := 40
def apple_quantity : ℕ := 10
def apple_price : ℚ := 30
def grape_discount : ℚ := 0.1
def mango_tax : ℚ := 0.05

theorem total_amount_paid : 
  (grape_quantity * grape_price * (1 - grape_discount) + 
   mango_quantity * mango_price * (1 + mango_tax) + 
   orange_quantity * orange_price + 
   apple_quantity * apple_price) = 1466 := by
  sorry

end total_amount_paid_l212_21269


namespace factorization_proof_l212_21257

theorem factorization_proof (y : ℝ) : 4*y*(y+2) + 9*(y+2) + 2*(y+2) = (y+2)*(4*y+11) := by
  sorry

end factorization_proof_l212_21257


namespace ice_cost_l212_21279

theorem ice_cost (cost_two_bags : ℝ) (num_bags : ℕ) : 
  cost_two_bags = 1.46 → num_bags = 4 → num_bags * (cost_two_bags / 2) = 2.92 := by
  sorry

end ice_cost_l212_21279


namespace connie_marbles_l212_21294

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 593

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776 := by
  sorry

end connie_marbles_l212_21294


namespace symmetry_about_a_periodicity_l212_21280

variable (f : ℝ → ℝ)
variable (a b : ℝ)

axiom a_nonzero : a ≠ 0
axiom b_diff_a : b ≠ a
axiom f_symmetry : ∀ x, f (a + x) = f (a - x)

theorem symmetry_about_a : ∀ x, f x = f (2*a - x) := by sorry

axiom symmetry_about_b : ∀ x, f x = f (2*b - x)

theorem periodicity : ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x := by sorry

end symmetry_about_a_periodicity_l212_21280


namespace calculation_proof_l212_21205

theorem calculation_proof : 
  71 * ((5 + 2/7) - (6 + 1/3)) / ((3 + 1/2) + (2 + 1/5)) = -(13 + 37/1197) := by
  sorry

end calculation_proof_l212_21205


namespace simplify_expression_l212_21252

theorem simplify_expression (x : ℝ) (hx : x ≥ 0) : (3 * x^(1/2))^6 = 729 * x^3 := by
  sorry

end simplify_expression_l212_21252


namespace cube_root_problem_l212_21233

theorem cube_root_problem :
  ∀ (a b : ℤ) (c : ℚ),
  (5 * a - 2 : ℚ) = -27 →
  b = ⌊Real.sqrt 22⌋ →
  c = -(4 / 25 : ℚ).sqrt →
  a = -5 ∧
  b = 4 ∧
  c = -2/5 ∧
  Real.sqrt (4 * (a : ℚ) * c + 7 * (b : ℚ)) = 6 :=
by sorry

end cube_root_problem_l212_21233


namespace total_population_avalon_l212_21235

theorem total_population_avalon (num_towns : ℕ) (avg_lower avg_upper : ℝ) :
  num_towns = 25 →
  5400 ≤ avg_lower →
  avg_upper ≤ 5700 →
  avg_lower ≤ (avg_lower + avg_upper) / 2 →
  (avg_lower + avg_upper) / 2 ≤ avg_upper →
  ∃ (total_population : ℝ),
    total_population = num_towns * ((avg_lower + avg_upper) / 2) ∧
    total_population = 138750 :=
by sorry

end total_population_avalon_l212_21235


namespace perpendicular_sufficient_not_necessary_l212_21255

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_sufficient_not_necessary
  (l m : Line) (α : Plane)
  (h_different : l ≠ m)
  (h_parallel : parallel m α) :
  (∀ l m α, perpendicular_to_plane l α → perpendicular l m) ∧
  (∃ l m α, perpendicular l m ∧ ¬ perpendicular_to_plane l α) :=
sorry

end perpendicular_sufficient_not_necessary_l212_21255


namespace square_of_binomial_l212_21243

theorem square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16*x^2 + 40*x + a = (4*x + b)^2) → a = 25 := by
sorry

end square_of_binomial_l212_21243


namespace OTVSU_shape_l212_21254

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The figure OTVSU -/
structure OTVSU where
  O : Point2D
  T : Point2D
  V : Point2D
  S : Point2D
  U : Point2D

/-- Predicate to check if a figure is a parallelogram -/
def isParallelogram (f : OTVSU) : Prop := sorry

/-- Predicate to check if a figure is a straight line -/
def isStraightLine (f : OTVSU) : Prop := sorry

/-- Predicate to check if a figure is a trapezoid -/
def isTrapezoid (f : OTVSU) : Prop := sorry

theorem OTVSU_shape :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  let f : OTVSU := {
    O := ⟨0, 0⟩,
    T := ⟨x₁, y₁⟩,
    V := ⟨x₁ + x₂, y₁ + y₂⟩,
    S := ⟨x₁ - x₂, y₁ - y₂⟩,
    U := ⟨x₂, y₂⟩
  }
  (isParallelogram f ∨ isStraightLine f) ∧ ¬isTrapezoid f := by
  sorry

end OTVSU_shape_l212_21254


namespace ellipse_intersection_theorem_l212_21222

/-- The ellipse C in standard form -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line l that intersects the ellipse -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- The condition for the intersection points A and B -/
def intersection_condition (xA yA xB yB : ℝ) : Prop :=
  (2 * xA + xB)^2 + (2 * yA + yB)^2 = (2 * xA - xB)^2 + (2 * yA - yB)^2

/-- The main theorem -/
theorem ellipse_intersection_theorem :
  ∀ (k m : ℝ),
    (∃ (xA yA xB yB : ℝ),
      ellipse_C xA yA ∧ ellipse_C xB yB ∧
      line_l k m xA yA ∧ line_l k m xB yB ∧
      intersection_condition xA yA xB yB) ↔
    (m < -Real.sqrt 3 / 2 ∨ m > Real.sqrt 3 / 2) :=
sorry

end ellipse_intersection_theorem_l212_21222


namespace sqrt_difference_comparison_l212_21263

theorem sqrt_difference_comparison 
  (a b m : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hm : m > 0) 
  (hab : a > b) : 
  Real.sqrt (b + m) - Real.sqrt b > Real.sqrt (a + m) - Real.sqrt a := by
  sorry

end sqrt_difference_comparison_l212_21263


namespace second_worker_de_time_l212_21239

/-- Represents a worker paving a path -/
structure Worker where
  speed : ℝ
  distance : ℝ

/-- Represents the paving scenario -/
structure PavingScenario where
  worker1 : Worker
  worker2 : Worker
  totalTime : ℝ

/-- The theorem statement -/
theorem second_worker_de_time (scenario : PavingScenario) : 
  scenario.worker1.speed > 0 ∧ 
  scenario.worker2.speed = 1.2 * scenario.worker1.speed ∧
  scenario.totalTime = 9 ∧
  scenario.worker1.distance * scenario.worker2.speed = scenario.worker2.distance * scenario.worker1.speed →
  ∃ (de_time : ℝ), de_time = 45 ∧ de_time = (scenario.totalTime * 60) / 12 :=
by sorry

end second_worker_de_time_l212_21239


namespace recurrence_sequence_a8_l212_21271

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a n + a (n + 1))

theorem recurrence_sequence_a8 
  (a : ℕ → ℕ) 
  (h : RecurrenceSequence a) 
  (h7 : a 7 = 120) : 
  a 8 = 194 := by
sorry

end recurrence_sequence_a8_l212_21271


namespace consecutive_integers_square_sum_l212_21200

theorem consecutive_integers_square_sum (a b c d e : ℕ) : 
  a > 0 → 
  b = a + 1 → 
  c = a + 2 → 
  d = a + 3 → 
  e = a + 4 → 
  a^2 + b^2 + c^2 = d^2 + e^2 → 
  a = 10 := by
sorry

end consecutive_integers_square_sum_l212_21200


namespace window_purchase_savings_l212_21229

/-- Calculates the cost of purchasing windows with a discount after the first five -/
def calculateCost (regularPrice : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  if quantity ≤ 5 then
    regularPrice * quantity
  else
    regularPrice * 5 + (regularPrice - discount) * (quantity - 5)

theorem window_purchase_savings :
  let regularPrice : ℕ := 120
  let discount : ℕ := 20
  let daveWindows : ℕ := 10
  let dougWindows : ℕ := 13
  let daveCost := calculateCost regularPrice discount daveWindows
  let dougCost := calculateCost regularPrice discount dougWindows
  let jointCost := calculateCost regularPrice discount (daveWindows + dougWindows)
  daveCost + dougCost - jointCost = 100 := by sorry

end window_purchase_savings_l212_21229


namespace emmy_and_gerry_apples_l212_21247

/-- The number of apples that can be bought with a given amount of money at a given price per apple -/
def apples_buyable (money : ℕ) (price : ℕ) : ℕ :=
  money / price

theorem emmy_and_gerry_apples : 
  let apple_price : ℕ := 2
  let emmy_money : ℕ := 200
  let gerry_money : ℕ := 100
  apples_buyable emmy_money apple_price + apples_buyable gerry_money apple_price = 150 :=
by sorry

end emmy_and_gerry_apples_l212_21247


namespace symmetric_point_x_axis_example_l212_21291

/-- The point symmetric to (x, y) with respect to the x-axis is (x, -y) -/
def symmetricPointXAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The statement that the point symmetric to (-1, 2) with respect to the x-axis is (-1, -2) -/
theorem symmetric_point_x_axis_example : symmetricPointXAxis (-1, 2) = (-1, -2) := by
  sorry

end symmetric_point_x_axis_example_l212_21291


namespace problem_one_problem_two_l212_21220

-- Problem 1
theorem problem_one : Real.sqrt 9 - (-2023)^(0 : ℤ) + 2^(-1 : ℤ) = 5/2 := by sorry

-- Problem 2
theorem problem_two (a b : ℝ) (hb : b ≠ 0) :
  (a / b - 1) / ((a^2 - b^2) / (2*b)) = 2 / (a + b) := by sorry

end problem_one_problem_two_l212_21220


namespace parabola_x_intercept_difference_l212_21225

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-value for a given x-value in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a quadratic function -/
def QuadraticFunction.contains_point (f : QuadraticFunction) (p : Point) : Prop :=
  f.eval p.x = p.y

/-- Calculates the difference between the x-intercepts of a quadratic function -/
noncomputable def x_intercept_difference (f : QuadraticFunction) : ℝ :=
  sorry

theorem parabola_x_intercept_difference :
  ∀ (f : QuadraticFunction),
  (∃ (v : Point), v.x = 3 ∧ v.y = -9 ∧ f.contains_point v) →
  f.contains_point ⟨5, 7⟩ →
  x_intercept_difference f = 3 := by
  sorry

end parabola_x_intercept_difference_l212_21225


namespace range_of_a_l212_21213

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

-- Define the set of valid values for a
def valid_a : Set ℝ := {a | (1 < a ∧ a < 2) ∨ a ≤ -2}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ valid_a :=
by sorry

end range_of_a_l212_21213


namespace sum_of_digits_equals_four_l212_21216

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5^i)) 0

/-- Converts a decimal number to base-6 --/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The base-5 representation of 2014₅ --/
def base5Number : List Nat := [4, 1, 0, 2]

theorem sum_of_digits_equals_four :
  let decimal := base5ToDecimal base5Number
  let base6 := decimalToBase6 decimal
  base6.sum = 4 := by sorry

end sum_of_digits_equals_four_l212_21216


namespace magic_square_x_value_l212_21295

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ
  sum_eq : a + b + c = d + e + f ∧ 
           a + b + c = g + h + i ∧ 
           a + b + c = a + d + g ∧ 
           a + b + c = b + e + h ∧ 
           a + b + c = c + f + i ∧ 
           a + b + c = a + e + i ∧ 
           a + b + c = c + e + g

/-- Theorem stating that x must be 230 in the given magic square -/
theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.a = x)
  (h2 : ms.b = 25)
  (h3 : ms.c = 110)
  (h4 : ms.d = 5) :
  x = 230 := by
  sorry

end magic_square_x_value_l212_21295


namespace pythagorean_sum_and_difference_squares_l212_21238

theorem pythagorean_sum_and_difference_squares (a b c : ℕ+) 
  (h : c^2 = a^2 + b^2) : 
  ∃ (x y z w : ℕ+), c^2 + a*b = x^2 + y^2 ∧ c^2 - a*b = z^2 + w^2 := by
  sorry

end pythagorean_sum_and_difference_squares_l212_21238


namespace soda_cost_calculation_l212_21245

theorem soda_cost_calculation (total_cost sandwich_cost : ℚ) 
  (num_sandwiches num_sodas : ℕ) :
  total_cost = 6.46 →
  sandwich_cost = 1.49 →
  num_sandwiches = 2 →
  num_sodas = 4 →
  (total_cost - (↑num_sandwiches * sandwich_cost)) / ↑num_sodas = 0.87 := by
  sorry

end soda_cost_calculation_l212_21245


namespace correct_propositions_count_l212_21299

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) : Line := sorry
def skew_lines (l1 l2 : Line) : Prop := sorry

-- Define the propositions
def proposition1 (m n : Line) (α : Plane) : Prop :=
  perpendicular_lines m n → perpendicular m α → parallel n α

def proposition2 (m n : Line) (α β : Plane) : Prop :=
  perpendicular m α → perpendicular n β → parallel_lines m n → parallel_planes α β

def proposition3 (m n : Line) (α β : Plane) : Prop :=
  skew_lines m n → line_in_plane m α → line_in_plane n β → parallel m β → parallel n α → parallel_planes α β

def proposition4 (m n : Line) (α β : Plane) : Prop :=
  perpendicular_planes α β → intersection α β = m → line_in_plane n β → perpendicular_lines n m → perpendicular n α

-- Theorem statement
theorem correct_propositions_count :
  ¬proposition1 m n α ∧
  proposition2 m n α β ∧
  proposition3 m n α β ∧
  proposition4 m n α β :=
sorry

end correct_propositions_count_l212_21299


namespace sum_of_roots_lower_bound_l212_21250

theorem sum_of_roots_lower_bound (k : ℝ) (α β : ℝ) : 
  (∃ x : ℝ, x^2 - 2*(1-k)*x + k^2 = 0) →
  (α^2 - 2*(1-k)*α + k^2 = 0) →
  (β^2 - 2*(1-k)*β + k^2 = 0) →
  α + β ≥ 1 := by sorry

end sum_of_roots_lower_bound_l212_21250


namespace house_rent_expenditure_l212_21264

/-- Given a person's income and expenditure pattern, calculate their house rent expense -/
theorem house_rent_expenditure (income : ℝ) (petrol_expense : ℝ) :
  petrol_expense = 0.3 * income →
  petrol_expense = 300 →
  let remaining_income := income - petrol_expense
  let house_rent := 0.2 * remaining_income
  house_rent = 140 := by
  sorry

end house_rent_expenditure_l212_21264


namespace max_real_sum_l212_21289

/-- The zeroes of z^10 - 2^30 -/
def zeroes : Finset ℂ :=
  sorry

/-- A function that chooses either z or iz to maximize the real part -/
def w (z : ℂ) : ℂ :=
  sorry

/-- The sum of w(z) for all zeroes -/
def sum_w : ℂ :=
  sorry

/-- The maximum possible value of the real part of the sum -/
theorem max_real_sum :
  (sum_w.re : ℝ) = 16 * (1 + Real.cos (π / 5) + Real.cos (2 * π / 5) - Real.sin (3 * π / 5) - Real.sin (4 * π / 5)) :=
sorry

end max_real_sum_l212_21289


namespace existence_of_prime_and_sequence_l212_21240

theorem existence_of_prime_and_sequence (k : ℕ+) :
  ∃ (p : ℕ) (a : Fin (k+3) → ℕ), 
    Prime p ∧ 
    (∀ i : Fin (k+3), 1 ≤ a i ∧ a i < p) ∧
    (∀ i j : Fin (k+3), i ≠ j → a i ≠ a j) ∧
    (∀ i : Fin k, p ∣ (a i * a (i+1) * a (i+2) * a (i+3) - i)) :=
by sorry

end existence_of_prime_and_sequence_l212_21240


namespace min_value_ab_l212_21212

theorem min_value_ab (a b : ℝ) (h : 0 < a ∧ 0 < b) (eq : 1/a + 2/b = Real.sqrt (a*b)) : 
  2 * Real.sqrt 2 ≤ a * b := by
  sorry

end min_value_ab_l212_21212


namespace janets_group_children_count_l212_21292

theorem janets_group_children_count 
  (total_people : Nat) 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (discount_rate : ℚ) 
  (soda_price : ℚ) 
  (total_paid : ℚ) :
  total_people = 10 ∧ 
  adult_price = 30 ∧ 
  child_price = 15 ∧ 
  discount_rate = 0.8 ∧ 
  soda_price = 5 ∧ 
  total_paid = 197 →
  ∃ (children : Nat),
    children ≤ total_people ∧
    (total_paid - soda_price) = 
      ((adult_price * (total_people - children) + child_price * children) * discount_rate) ∧
    children = 4 := by
  sorry

end janets_group_children_count_l212_21292


namespace group_size_calculation_l212_21285

theorem group_size_calculation (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 2.5 ∧ old_weight = 65 ∧ new_weight = 90 →
  ∃ n : ℕ, n = 10 ∧ n * average_increase = new_weight - old_weight :=
by
  sorry

end group_size_calculation_l212_21285


namespace car_travel_time_l212_21218

/-- Proves that a car traveling 810 km at 162 km/h takes 5 hours -/
theorem car_travel_time (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 810 ∧ speed = 162 → time = distance / speed → time = 5 := by
  sorry

end car_travel_time_l212_21218


namespace student_allowance_l212_21241

theorem student_allowance (initial_allowance : ℚ) : 
  let remaining_after_clothes := initial_allowance * (4/7)
  let remaining_after_games := remaining_after_clothes * (3/5)
  let remaining_after_books := remaining_after_games * (5/9)
  let remaining_after_donation := remaining_after_books * (1/2)
  remaining_after_donation = 3.75 → initial_allowance = 39.375 := by
  sorry

end student_allowance_l212_21241


namespace correct_mean_calculation_l212_21215

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 ∧ 
  initial_mean = 250 ∧ 
  incorrect_value = 135 ∧ 
  correct_value = 165 →
  (n * initial_mean + (correct_value - incorrect_value)) / n = 251 := by
  sorry

end correct_mean_calculation_l212_21215


namespace kim_average_unchanged_l212_21203

def kim_scores : List ℝ := [92, 86, 95, 89, 93]

theorem kim_average_unchanged (scores := kim_scores) :
  let first_three_avg := (scores.take 3).sum / 3
  let all_five_avg := scores.sum / 5
  all_five_avg - first_three_avg = 0 := by
sorry

end kim_average_unchanged_l212_21203


namespace parallelogram_side_lengths_l212_21204

def parallelogram_properties (angle : ℝ) (shorter_diagonal : ℝ) (perpendicular : ℝ) : Prop :=
  angle = 60 ∧ 
  shorter_diagonal = 2 * Real.sqrt 31 ∧ 
  perpendicular = Real.sqrt 75 / 2

theorem parallelogram_side_lengths 
  (angle : ℝ) (shorter_diagonal : ℝ) (perpendicular : ℝ) 
  (h : parallelogram_properties angle shorter_diagonal perpendicular) :
  ∃ (longer_side shorter_side longer_diagonal : ℝ),
    longer_side = 12 ∧ 
    shorter_side = 10 ∧ 
    longer_diagonal = 2 * Real.sqrt 91 :=
by sorry

end parallelogram_side_lengths_l212_21204


namespace value_of_b_l212_21223

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 3) : b = 3 := by
  sorry

end value_of_b_l212_21223


namespace sum_of_digits_M_l212_21267

/-- The third smallest positive integer divisible by all integers less than 9 -/
def M : ℕ := sorry

/-- M is divisible by all positive integers less than 9 -/
axiom M_divisible (n : ℕ) (h : n > 0 ∧ n < 9) : M % n = 0

/-- M is the third smallest such integer -/
axiom M_third_smallest :
  ∀ k : ℕ, k > 0 ∧ k < M → (∀ n : ℕ, n > 0 ∧ n < 9 → k % n = 0) →
  ∃ j : ℕ, j > 0 ∧ j < M ∧ j ≠ k ∧ (∀ n : ℕ, n > 0 ∧ n < 9 → j % n = 0)

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem sum_of_digits_M : sum_of_digits M = 9 := sorry

end sum_of_digits_M_l212_21267


namespace perpendicular_vectors_imply_m_value_l212_21221

/-- Given vectors a, b, and c in ℝ², prove that if a - b is perpendicular to c,
    then the value of m in b is -3. -/
theorem perpendicular_vectors_imply_m_value (a b c : ℝ × ℝ) (m : ℝ) :
  a = (-1, 2) →
  b = (m, -1) →
  c = (3, -2) →
  (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0 →
  m = -3 := by
  sorry

#check perpendicular_vectors_imply_m_value

end perpendicular_vectors_imply_m_value_l212_21221


namespace function_correspondence_l212_21242

-- Case 1
def A1 : Set ℕ := {1, 2, 3}
def B1 : Set ℕ := {7, 8, 9}
def f1 : ℕ → ℕ
  | 1 => 7
  | 2 => 7
  | 3 => 8
  | _ => 0  -- default case for completeness

-- Case 2
def A2 : Set ℕ := {1, 2, 3}
def B2 : Set ℕ := {1, 2, 3}
def f2 : ℕ → ℕ
  | x => 2 * x - 1

-- Case 3
def A3 : Set ℝ := {x : ℝ | x ≥ -1}
def B3 : Set ℝ := {x : ℝ | x ≥ -1}
def f3 : ℝ → ℝ
  | x => 2 * x + 1

-- Case 4
def A4 : Set ℤ := Set.univ
def B4 : Set ℤ := {-1, 1}
def f4 : ℤ → ℤ
  | n => if n % 2 = 0 then 1 else -1

theorem function_correspondence :
  (∀ x ∈ A1, f1 x ∈ B1) ∧
  (¬∀ x ∈ A2, f2 x ∈ B2) ∧
  (∀ x ∈ A3, f3 x ∈ B3) ∧
  (∀ x ∈ A4, f4 x ∈ B4) :=
by sorry

end function_correspondence_l212_21242


namespace coordinate_sum_of_point_B_l212_21272

/-- Given points A(0, 0) and B(x, 5) where the slope of AB is 3/4,
    prove that the sum of x- and y-coordinates of B is 35/3 -/
theorem coordinate_sum_of_point_B (x : ℚ) : 
  let A : ℚ × ℚ := (0, 0)
  let B : ℚ × ℚ := (x, 5)
  let slope : ℚ := (B.2 - A.2) / (B.1 - A.1)
  slope = 3/4 → x + 5 = 35/3 := by
  sorry

end coordinate_sum_of_point_B_l212_21272


namespace line_through_point_l212_21234

/-- Theorem: If the line ax + 3y - 2 = 0 passes through point (1, 0), then a = 2. -/
theorem line_through_point (a : ℝ) : 
  (∀ x y, a * x + 3 * y - 2 = 0 → (x = 1 ∧ y = 0)) → a = 2 :=
by sorry

end line_through_point_l212_21234


namespace collinear_points_k_value_l212_21297

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- Given three collinear points (4, 10), (-3, k), and (-8, 5), prove that k = 85/12. -/
theorem collinear_points_k_value :
  collinear 4 10 (-3) k (-8) 5 → k = 85/12 :=
by
  sorry


end collinear_points_k_value_l212_21297


namespace find_x_l212_21214

theorem find_x : ∃ x : ℝ, 5.76 = 0.12 * (0.40 * x) ∧ x = 120 := by
  sorry

end find_x_l212_21214


namespace divisibility_sequence_l212_21283

theorem divisibility_sequence (a : ℕ) : ∃ n : ℕ, ∀ k : ℕ, a ∣ (n^(n^k) + 1) := by
  sorry

end divisibility_sequence_l212_21283


namespace freddy_is_18_l212_21209

def job_age : ℕ := 5

def stephanie_age (j : ℕ) : ℕ := 4 * j

def freddy_age (s : ℕ) : ℕ := s - 2

theorem freddy_is_18 : freddy_age (stephanie_age job_age) = 18 := by
  sorry

end freddy_is_18_l212_21209


namespace square_between_bounds_l212_21276

theorem square_between_bounds (n : ℕ) (hn : n ≥ 16088121) :
  ∃ l : ℕ, n < l ^ 2 ∧ l ^ 2 < n * (1 + 1 / 2005) := by
  sorry

end square_between_bounds_l212_21276


namespace line_tangent_to_circle_l212_21284

/-- The line 5x + 12y + a = 0 is tangent to the circle (x-1)^2 + y^2 = 1 if and only if a = -18 or a = 8 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, 5*x + 12*y + a = 0 → (x-1)^2 + y^2 = 1) ↔ (a = -18 ∨ a = 8) := by
  sorry

end line_tangent_to_circle_l212_21284


namespace sum_of_cubes_up_to_8_l212_21286

/-- Sum of cubes from 1³ to n³ equals the square of the sum of first n natural numbers -/
def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

/-- The sum of cubes from 1³ to 8³ is 1296 -/
theorem sum_of_cubes_up_to_8 : sum_of_cubes 8 = 1296 := by
  sorry

end sum_of_cubes_up_to_8_l212_21286


namespace tan_alpha_value_l212_21290

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 5) :
  Real.tan α = -27/14 := by
  sorry

end tan_alpha_value_l212_21290


namespace specific_trapezoid_area_l212_21273

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- The length of the shorter base of the trapezoid -/
  shorterBase : ℝ
  /-- The length of the longer segment of the non-parallel side divided by the point of tangency -/
  longerSegment : ℝ
  /-- The length of the shorter segment of the non-parallel side divided by the point of tangency -/
  shorterSegment : ℝ
  /-- The shorter base is positive -/
  shorterBase_pos : 0 < shorterBase
  /-- The longer segment is positive -/
  longerSegment_pos : 0 < longerSegment
  /-- The shorter segment is positive -/
  shorterSegment_pos : 0 < shorterSegment

/-- The area of the trapezoid -/
def area (t : InscribedCircleTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the specific trapezoid is 198 -/
theorem specific_trapezoid_area :
  ∀ t : InscribedCircleTrapezoid,
  t.shorterBase = 6 ∧ t.longerSegment = 9 ∧ t.shorterSegment = 4 →
  area t = 198 := by
  sorry

end specific_trapezoid_area_l212_21273


namespace smallest_fourth_number_l212_21260

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest two-digit positive integer that satisfies the given condition -/
theorem smallest_fourth_number : 
  ∃ x : ℕ, 
    x ≥ 10 ∧ x < 100 ∧ 
    (∀ y : ℕ, y ≥ 10 ∧ y < 100 → 
      sumOfDigits 28 + sumOfDigits 46 + sumOfDigits 59 + sumOfDigits y = (28 + 46 + 59 + y) / 4 →
      x ≤ y) ∧
    sumOfDigits 28 + sumOfDigits 46 + sumOfDigits 59 + sumOfDigits x = (28 + 46 + 59 + x) / 4 ∧
    x = 11 := by
  sorry

end smallest_fourth_number_l212_21260


namespace circle_area_from_circumference_l212_21275

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 
  (2 * π * r = 36 * π) → 
  (π * r^2 = 324 * π) := by
sorry

end circle_area_from_circumference_l212_21275


namespace circle_properties_l212_21217

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, 0)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2) :=
by sorry

end circle_properties_l212_21217


namespace same_terminal_side_angle_l212_21224

-- Define a function to normalize an angle to the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Theorem statement
theorem same_terminal_side_angle :
  normalizeAngle (-390) = 330 :=
sorry

end same_terminal_side_angle_l212_21224


namespace circle_radius_proof_l212_21246

theorem circle_radius_proof (num_pencils : ℕ) (pencil_length : ℚ) (inches_per_foot : ℕ) :
  num_pencils = 56 →
  pencil_length = 6 →
  inches_per_foot = 12 →
  (num_pencils * pencil_length / (2 * inches_per_foot) : ℚ) = 14 := by
  sorry

end circle_radius_proof_l212_21246


namespace unique_quadratic_function_l212_21262

/-- A quadratic function with a negative leading coefficient -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_neg : a < 0

/-- The function f(x) -/
def f (qf : QuadraticFunction) (x : ℝ) : ℝ := qf.a * x^2 + qf.b * x + qf.c

/-- The condition that 1 and 3 are roots of y = f(x) + 2x -/
def roots_condition (qf : QuadraticFunction) : Prop :=
  f qf 1 + 2 * 1 = 0 ∧ f qf 3 + 2 * 3 = 0

/-- The condition that f(x) + 6a = 0 has two equal roots -/
def equal_roots_condition (qf : QuadraticFunction) : Prop :=
  ∃ (x : ℝ), f qf x + 6 * qf.a = 0 ∧ 
  ∀ (y : ℝ), f qf y + 6 * qf.a = 0 → y = x

/-- The theorem statement -/
theorem unique_quadratic_function :
  ∃! (qf : QuadraticFunction),
    roots_condition qf ∧
    equal_roots_condition qf ∧
    qf.a = -1/4 ∧ qf.b = -1 ∧ qf.c = -3/4 :=
sorry

end unique_quadratic_function_l212_21262


namespace pizza_varieties_count_l212_21258

/-- Represents the number of base pizza flavors -/
def num_flavors : ℕ := 8

/-- Represents the number of extra topping options -/
def num_toppings : ℕ := 5

/-- Calculates the number of valid topping combinations -/
def valid_topping_combinations : ℕ :=
  (num_toppings) +  -- 1 topping
  (num_toppings.choose 2 - 1) +  -- 2 toppings, excluding onions with jalapeños
  (num_toppings.choose 3 - 3)  -- 3 toppings, excluding combinations with both onions and jalapeños

/-- The total number of pizza varieties -/
def total_varieties : ℕ := num_flavors * valid_topping_combinations

theorem pizza_varieties_count :
  total_varieties = 168 := by sorry

end pizza_varieties_count_l212_21258


namespace village_population_problem_l212_21253

theorem village_population_problem (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.85 = 2907 → P = 3801 := by
  sorry

end village_population_problem_l212_21253


namespace hair_length_calculation_l212_21211

/-- Calculates the final hair length after a series of changes. -/
def finalHairLength (initialLength : ℝ) (firstCutFraction : ℝ) (growth : ℝ) (secondCut : ℝ) : ℝ :=
  (initialLength - firstCutFraction * initialLength + growth) - secondCut

/-- Theorem stating that given the specific hair length changes, the final length is 14 inches. -/
theorem hair_length_calculation :
  finalHairLength 24 0.5 4 2 = 14 := by
  sorry

end hair_length_calculation_l212_21211
