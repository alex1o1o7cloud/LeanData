import Mathlib

namespace NUMINAMATH_CALUDE_gizmo_production_l1339_133943

/-- Represents the production rate of gadgets per worker per hour -/
def gadget_rate : ℝ := 2

/-- Represents the production rate of gizmos per worker per hour -/
def gizmo_rate : ℝ := 1.5

/-- Represents the number of workers -/
def workers : ℕ := 40

/-- Represents the total working hours -/
def total_hours : ℝ := 6

/-- Represents the number of gadgets to be produced -/
def gadgets_to_produce : ℕ := 240

theorem gizmo_production :
  let hours_for_gadgets : ℝ := gadgets_to_produce / (workers * gadget_rate)
  let remaining_hours : ℝ := total_hours - hours_for_gadgets
  ↑workers * gizmo_rate * remaining_hours = 180 :=
sorry

end NUMINAMATH_CALUDE_gizmo_production_l1339_133943


namespace NUMINAMATH_CALUDE_total_seats_calculation_l1339_133930

/-- The number of students per bus -/
def students_per_bus : ℝ := 14.0

/-- The number of buses -/
def number_of_buses : ℝ := 2.0

/-- The total number of seats taken up by students -/
def total_seats : ℝ := students_per_bus * number_of_buses

theorem total_seats_calculation : total_seats = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_seats_calculation_l1339_133930


namespace NUMINAMATH_CALUDE_evaluate_expression_l1339_133950

theorem evaluate_expression : 7 - 5 * (9 - 4^2) * 3 = 112 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1339_133950


namespace NUMINAMATH_CALUDE_correct_additional_muffins_l1339_133928

/-- Calculates the additional muffins needed for a charity event -/
def additional_muffins_needed (target : ℕ) (arthur_baked : ℕ) (beatrice_baked : ℕ) (charles_baked : ℕ) : ℕ :=
  target - (arthur_baked + beatrice_baked + charles_baked)

/-- Proves the correctness of additional muffins calculations for three charity events -/
theorem correct_additional_muffins :
  (additional_muffins_needed 200 35 48 29 = 88) ∧
  (additional_muffins_needed 150 20 35 25 = 70) ∧
  (additional_muffins_needed 250 45 60 30 = 115) := by
  sorry

#eval additional_muffins_needed 200 35 48 29
#eval additional_muffins_needed 150 20 35 25
#eval additional_muffins_needed 250 45 60 30

end NUMINAMATH_CALUDE_correct_additional_muffins_l1339_133928


namespace NUMINAMATH_CALUDE_amy_cupcake_packages_l1339_133983

def cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package

theorem amy_cupcake_packages :
  cupcake_packages 50 5 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_amy_cupcake_packages_l1339_133983


namespace NUMINAMATH_CALUDE_product_representation_l1339_133964

theorem product_representation (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  ∃ p q : ℝ, (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) = p + q * Real.sqrt 5 ∧ p ≥ 0 ∧ q ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_representation_l1339_133964


namespace NUMINAMATH_CALUDE_pet_store_birds_l1339_133942

/-- Calculates the total number of birds in a pet store given the number of cages and birds per cage. -/
def total_birds (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) : ℕ :=
  num_cages * (parrots_per_cage + parakeets_per_cage)

/-- Proves that the pet store has 72 birds in total. -/
theorem pet_store_birds : total_birds 9 2 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l1339_133942


namespace NUMINAMATH_CALUDE_deposit_percentage_l1339_133918

/-- Proves that the percentage P of the initial amount used in the deposit calculation is 30% --/
theorem deposit_percentage (initial_amount deposit_amount : ℝ) 
  (h1 : initial_amount = 50000)
  (h2 : deposit_amount = 750)
  (h3 : ∃ P : ℝ, deposit_amount = 0.20 * 0.25 * (P / 100) * initial_amount) :
  ∃ P : ℝ, P = 30 ∧ deposit_amount = 0.20 * 0.25 * (P / 100) * initial_amount :=
by sorry

end NUMINAMATH_CALUDE_deposit_percentage_l1339_133918


namespace NUMINAMATH_CALUDE_playground_children_count_l1339_133991

theorem playground_children_count : 
  let boys : ℕ := 27
  let girls : ℕ := 35
  let total_children : ℕ := boys + girls
  total_children = 62 := by
  sorry

end NUMINAMATH_CALUDE_playground_children_count_l1339_133991


namespace NUMINAMATH_CALUDE_terms_before_negative_three_l1339_133956

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem terms_before_negative_three (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 105 ∧ d = -6 →
  (∀ k < n, arithmetic_sequence a₁ d k > -3) ∧
  arithmetic_sequence a₁ d n = -3 →
  n - 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_terms_before_negative_three_l1339_133956


namespace NUMINAMATH_CALUDE_weather_forecast_probability_meaning_l1339_133975

/-- Represents the probability of an event -/
def Probability := ℝ

/-- Represents a weather forecast statement -/
structure WeatherForecast where
  statement : String
  probability : Probability

/-- Represents the meaning of a probability statement -/
inductive ProbabilityMeaning
  | Possibility
  | TimePercentage
  | AreaPercentage
  | PeopleOpinion

/-- The correct interpretation of a probability statement in a weather forecast -/
def correct_interpretation : ProbabilityMeaning := ProbabilityMeaning.Possibility

/-- 
  Theorem: The statement "The probability of rain tomorrow in this area is 80%" 
  in a weather forecast means "The possibility of rain tomorrow in this area is 80%"
-/
theorem weather_forecast_probability_meaning 
  (forecast : WeatherForecast) 
  (h : forecast.statement = "The probability of rain tomorrow in this area is 80%") :
  correct_interpretation = ProbabilityMeaning.Possibility := by sorry

end NUMINAMATH_CALUDE_weather_forecast_probability_meaning_l1339_133975


namespace NUMINAMATH_CALUDE_count_tricycles_l1339_133952

/-- The number of tricycles in a bike shop, given the number of bicycles,
    the number of wheels per bicycle and tricycle, and the total number of wheels. -/
theorem count_tricycles (num_bicycles : ℕ) (wheels_per_bicycle : ℕ) (wheels_per_tricycle : ℕ) 
    (total_wheels : ℕ) (h1 : num_bicycles = 50) (h2 : wheels_per_bicycle = 2) 
    (h3 : wheels_per_tricycle = 3) (h4 : total_wheels = 160) : 
    (total_wheels - num_bicycles * wheels_per_bicycle) / wheels_per_tricycle = 20 := by
  sorry

end NUMINAMATH_CALUDE_count_tricycles_l1339_133952


namespace NUMINAMATH_CALUDE_work_completion_time_l1339_133920

theorem work_completion_time (a b c : ℝ) 
  (h1 : a + b + c = 1/4)  -- Combined work rate
  (h2 : a = 1/12)         -- a's work rate
  (h3 : b = 1/18)         -- b's work rate
  : c = 1/9 :=            -- c's work rate (to be proved)
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1339_133920


namespace NUMINAMATH_CALUDE_people_after_increase_l1339_133980

-- Define the initial conditions
def initial_people : ℕ := 5
def initial_houses : ℕ := 5
def initial_days : ℕ := 5

-- Define the new conditions
def new_houses : ℕ := 100
def new_days : ℕ := 5

-- Define the function to calculate the number of people needed
def people_needed (houses : ℕ) (days : ℕ) : ℕ :=
  (houses * initial_people * initial_days) / (initial_houses * days)

-- Theorem statement
theorem people_after_increase :
  people_needed new_houses new_days = 100 :=
by sorry

end NUMINAMATH_CALUDE_people_after_increase_l1339_133980


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l1339_133902

theorem tangent_equation_solution (x : Real) :
  5.30 * Real.tan x * Real.tan (20 * π / 180) +
  Real.tan (20 * π / 180) * Real.tan (40 * π / 180) +
  Real.tan (40 * π / 180) * Real.tan x = 1 →
  ∃ k : ℤ, x = (30 + 180 * k) * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l1339_133902


namespace NUMINAMATH_CALUDE_abs_is_even_and_decreasing_l1339_133911

def f (x : ℝ) := abs x

theorem abs_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_abs_is_even_and_decreasing_l1339_133911


namespace NUMINAMATH_CALUDE_xy_sum_eleven_l1339_133986

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem xy_sum_eleven (x y : ℝ) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hxy : x * y = 10) 
  (hineq : x^(log2 x) * y^(log2 y) ≥ 10) : 
  x + y = 11 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_eleven_l1339_133986


namespace NUMINAMATH_CALUDE_downstream_distance_man_downstream_distance_l1339_133988

/-- Calculates the downstream distance given swimming conditions --/
theorem downstream_distance (time : ℝ) (upstream_distance : ℝ) (still_speed : ℝ) : ℝ :=
  let stream_speed := still_speed - (upstream_distance / time)
  let downstream_speed := still_speed + stream_speed
  downstream_speed * time

/-- Proves that the downstream distance is 45 km given the specific conditions --/
theorem man_downstream_distance : 
  downstream_distance 5 25 7 = 45 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_man_downstream_distance_l1339_133988


namespace NUMINAMATH_CALUDE_division_problem_l1339_133922

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 12401 →
  divisor = 163 →
  remainder = 13 →
  dividend = divisor * quotient + remainder →
  quotient = 76 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1339_133922


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l1339_133901

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_b_value :
  ∃ b : ℝ, collinear 4 (-6) (b + 3) 4 (3*b - 2) 3 ∧ b = 17/7 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l1339_133901


namespace NUMINAMATH_CALUDE_alternating_arithmetic_series_sum_l1339_133990

def arithmetic_series (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => a₁ - i * d)

def alternating_sign (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => if i % 2 == 0 then 1 else -1)

def series_sum (series : List ℤ) : ℤ :=
  series.sum

theorem alternating_arithmetic_series_sum :
  let a₁ : ℤ := 2005
  let d : ℤ := 10
  let n : ℕ := 200
  let series := List.zip (arithmetic_series a₁ d n) (alternating_sign n) |>.map (fun (x, y) => x * y)
  series_sum series = 1000 := by
  sorry

end NUMINAMATH_CALUDE_alternating_arithmetic_series_sum_l1339_133990


namespace NUMINAMATH_CALUDE_root_sum_equals_three_l1339_133941

noncomputable section

-- Define the logarithm base 10 function
def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the equations for x₁ and x₂
def equation1 (x : ℝ) : Prop := x + log10 x = 3
def equation2 (x : ℝ) : Prop := x + 10^x = 3

-- State the theorem
theorem root_sum_equals_three 
  (x₁ x₂ : ℝ) 
  (h1 : equation1 x₁) 
  (h2 : equation2 x₂) : 
  x₁ + x₂ = 3 := by sorry

end

end NUMINAMATH_CALUDE_root_sum_equals_three_l1339_133941


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l1339_133913

/-- Sum of a geometric sequence -/
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem : geometric_sum (1/3) (1/2) 8 = 255/384 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l1339_133913


namespace NUMINAMATH_CALUDE_volume_of_S_l1339_133999

/-- A line in ℝ³ -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Distance from a point to a line in ℝ³ -/
def distPointToLine (p : ℝ × ℝ × ℝ) (l : Line3D) : ℝ :=
  sorry

/-- Distance between two points in ℝ³ -/
def distBetweenPoints (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The set S as described in the problem -/
def S (ℓ : Line3D) (P : ℝ × ℝ × ℝ) : Set (ℝ × ℝ × ℝ) :=
  {X | distPointToLine X ℓ ≥ 2 * distBetweenPoints X P}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

theorem volume_of_S (ℓ : Line3D) (P : ℝ × ℝ × ℝ) (d : ℝ) 
    (h_d : d > 0) (h_dist : distPointToLine P ℓ = d) :
    volume (S ℓ P) = (16 * Real.pi * d^3) / (27 * Real.sqrt 3) :=
  sorry

end NUMINAMATH_CALUDE_volume_of_S_l1339_133999


namespace NUMINAMATH_CALUDE_amount_less_than_five_times_number_l1339_133998

theorem amount_less_than_five_times_number (N : ℕ) (A : ℕ) : 
  N = 52 → A < 5 * N → A = 232 → A = A 
:= by sorry

end NUMINAMATH_CALUDE_amount_less_than_five_times_number_l1339_133998


namespace NUMINAMATH_CALUDE_horner_method_evaluation_l1339_133944

/-- Horner's Method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => a + x * acc) 0

/-- The polynomial function -/
def f (x : ℝ) : ℝ :=
  1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem horner_method_evaluation :
  let coeffs := [0.00833, 0.04167, 0.16667, 0.5, 1, 1]
  abs (horner_eval coeffs (-0.2) - f (-0.2)) < 1e-5 ∧
  abs (horner_eval coeffs (-0.2) - 0.00427) < 1e-5 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_evaluation_l1339_133944


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l1339_133961

theorem divisibility_by_seven (n a b d : ℤ) 
  (h1 : 0 ≤ b ∧ b ≤ 9)
  (h2 : 0 ≤ a)
  (h3 : n = 10 * a + b)
  (h4 : d = a - 2 * b) :
  7 ∣ n ↔ 7 ∣ d := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l1339_133961


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1339_133932

theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℝ) : 
  S n = 48 → S (2 * n) = 60 → S (3 * n) = 63 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1339_133932


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1339_133994

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) ↔ 
  (a < -2 ∨ a ≥ 6/5) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1339_133994


namespace NUMINAMATH_CALUDE_physical_education_marks_l1339_133949

def american_literature : ℕ := 66
def history : ℕ := 75
def home_economics : ℕ := 52
def art : ℕ := 89
def average_marks : ℕ := 70
def total_subjects : ℕ := 5

theorem physical_education_marks :
  let known_subjects_total := american_literature + history + home_economics + art
  let total_marks := average_marks * total_subjects
  total_marks - known_subjects_total = 68 := by sorry

end NUMINAMATH_CALUDE_physical_education_marks_l1339_133949


namespace NUMINAMATH_CALUDE_coin_difference_l1339_133909

def coin_values : List ℕ := [10, 20, 50]
def target_amount : ℕ := 145

def is_valid_combination (combination : List ℕ) : Prop :=
  combination.sum * 10 ≥ target_amount ∧
  (combination.sum * 10 - target_amount < 10 ∨ combination.sum * 10 = target_amount)

def min_coins : ℕ := sorry
def max_coins : ℕ := sorry

theorem coin_difference :
  ∃ (min_comb max_comb : List ℕ),
    is_valid_combination min_comb ∧
    is_valid_combination max_comb ∧
    min_comb.length = min_coins ∧
    max_comb.length = max_coins ∧
    max_coins - min_coins = 9 :=
  sorry

end NUMINAMATH_CALUDE_coin_difference_l1339_133909


namespace NUMINAMATH_CALUDE_remaining_family_member_age_l1339_133926

/-- Represents the ages of family members -/
structure FamilyAges where
  total : ℕ
  father : ℕ
  mother : ℕ
  brother : ℕ
  sister : ℕ
  remaining : ℕ

/-- Theorem stating the age of the remaining family member -/
theorem remaining_family_member_age 
  (family : FamilyAges)
  (h_total : family.total = 200)
  (h_father : family.father = 60)
  (h_mother : family.mother = family.father - 2)
  (h_brother : family.brother = family.father / 2)
  (h_sister : family.sister = 40)
  (h_sum : family.total = family.father + family.mother + family.brother + family.sister + family.remaining) :
  family.remaining = 12 :=
by sorry

end NUMINAMATH_CALUDE_remaining_family_member_age_l1339_133926


namespace NUMINAMATH_CALUDE_lower_limit_proof_l1339_133996

def is_prime (n : ℕ) : Prop := sorry

def count_primes_between (a b : ℝ) : ℕ := sorry

theorem lower_limit_proof : 
  ∀ x : ℕ, x ≤ 19 ↔ count_primes_between x (87/5) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_lower_limit_proof_l1339_133996


namespace NUMINAMATH_CALUDE_symmetry_about_point_period_four_l1339_133973

-- Define the function f
variable (f : ℝ → ℝ)

-- Statement ②
theorem symmetry_about_point (h : ∀ x, f (x + 1) + f (1 - x) = 0) :
  ∀ x, f (2 - x) = -f x :=
sorry

-- Statement ④
theorem period_four (h : ∀ x, f (1 + x) + f (x - 1) = 0) :
  ∀ x, f (x + 4) = f x :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_point_period_four_l1339_133973


namespace NUMINAMATH_CALUDE_triangle_cut_range_l1339_133937

/-- Given a triangle with side lengths 4, 5, and 6,
    if x is cut off from all sides resulting in an obtuse triangle,
    then 1 < x < 3 -/
theorem triangle_cut_range (x : ℝ) : 
  let a := 4 - x
  let b := 5 - x
  let c := 6 - x
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (a^2 + b^2 - c^2) / (2 * a * b) < 0 →
  1 < x ∧ x < 3 :=
by sorry


end NUMINAMATH_CALUDE_triangle_cut_range_l1339_133937


namespace NUMINAMATH_CALUDE_california_texas_plate_difference_l1339_133997

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^5 * num_digits^2

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_digits^2 * num_letters^4

/-- The difference in the number of possible license plates between California and Texas -/
def plate_difference : ℕ := california_plates - texas_plates

theorem california_texas_plate_difference :
  plate_difference = 1142440000 := by
  sorry

end NUMINAMATH_CALUDE_california_texas_plate_difference_l1339_133997


namespace NUMINAMATH_CALUDE_log_product_equality_l1339_133938

theorem log_product_equality : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l1339_133938


namespace NUMINAMATH_CALUDE_wooden_sticks_length_l1339_133955

/-- The total length of connected wooden sticks -/
def total_length (num_sticks : ℕ) (stick_length : ℕ) (overlap : ℕ) : ℕ :=
  num_sticks * stick_length - (num_sticks - 1) * overlap

/-- Proof that 6 wooden sticks of 50 cm each, connected with 10 cm overlaps, have a total length of 250 cm -/
theorem wooden_sticks_length :
  total_length 6 50 10 = 250 := by
  sorry

end NUMINAMATH_CALUDE_wooden_sticks_length_l1339_133955


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_g_l1339_133924

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

-- Define the function g
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem for the maximum value of g(x)
theorem max_value_g :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_g_l1339_133924


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1339_133976

theorem quadratic_root_problem (m : ℝ) :
  (∃ x : ℝ, x^2 + m*x + 6 = 0 ∧ x = -2) →
  (∃ x : ℝ, x^2 + m*x + 6 = 0 ∧ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1339_133976


namespace NUMINAMATH_CALUDE_jose_investment_is_4500_l1339_133959

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given shop investment scenario --/
def calculate_jose_investment (s : ShopInvestment) : ℕ :=
  -- The actual calculation is not implemented, as we only need the statement
  sorry

/-- Theorem stating that Jose's investment is 4500 given the problem conditions --/
theorem jose_investment_is_4500 (s : ShopInvestment) 
  (h1 : s.tom_investment = 3000)
  (h2 : s.jose_join_delay = 2)
  (h3 : s.total_profit = 5400)
  (h4 : s.jose_profit = 3000) :
  calculate_jose_investment s = 4500 := by
  sorry

#check jose_investment_is_4500

end NUMINAMATH_CALUDE_jose_investment_is_4500_l1339_133959


namespace NUMINAMATH_CALUDE_real_roots_condition_l1339_133972

theorem real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 2 * k * x + (k - 3) = 0) ↔ k ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_real_roots_condition_l1339_133972


namespace NUMINAMATH_CALUDE_rectangle_to_square_l1339_133917

theorem rectangle_to_square (k : ℕ) (n : ℕ) : 
  k > 7 →
  k * (k - 7) = n^2 →
  n < k →
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l1339_133917


namespace NUMINAMATH_CALUDE_solution_to_equation_l1339_133929

theorem solution_to_equation : ∃ (x y : ℝ), 2 * x - y = 5 ∧ x = 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1339_133929


namespace NUMINAMATH_CALUDE_solution_exists_iff_a_in_interval_l1339_133900

/-- The system of equations has a solution within the specified square if and only if
    a is in the given interval for some integer k. -/
theorem solution_exists_iff_a_in_interval :
  ∀ (a : ℝ), ∃ (x y : ℝ),
    (x * Real.sin a - y * Real.cos a = 2 * Real.sin a - Real.cos a) ∧
    (x - 3 * y + 13 = 0) ∧
    (5 ≤ x ∧ x ≤ 9) ∧
    (3 ≤ y ∧ y ≤ 7)
  ↔
    ∃ (k : ℤ), π/4 + k * π ≤ a ∧ a ≤ Real.arctan (5/3) + k * π :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_iff_a_in_interval_l1339_133900


namespace NUMINAMATH_CALUDE_expand_product_l1339_133985

theorem expand_product (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1339_133985


namespace NUMINAMATH_CALUDE_quadratic_root_shift_l1339_133933

theorem quadratic_root_shift (a b : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, x = 2021 ∧ a * x^2 + b * x + 2 = 0) →
  (∃ y : ℝ, y = 2022 ∧ a * (y - 1)^2 + b * (y - 1) = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_shift_l1339_133933


namespace NUMINAMATH_CALUDE_inequality_and_optimization_l1339_133965

theorem inequality_and_optimization (a b m n : ℝ) (x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n) 
  (hx : 0 < x ∧ x < 1/2) : 
  (m^2 / a + n^2 / b ≥ (m + n)^2 / (a + b)) ∧ 
  (∀ y ∈ Set.Ioo 0 (1/2), 2/x + 9/(1-2*x) ≤ 2/y + 9/(1-2*y)) ∧
  (2/x + 9/(1-2*x) = 25) ∧ (x = 1/5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_optimization_l1339_133965


namespace NUMINAMATH_CALUDE_determinant_equality_l1339_133979

theorem determinant_equality (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = -3 →
  Matrix.det !![p + 2*r, q + 2*s; r, s] = -3 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l1339_133979


namespace NUMINAMATH_CALUDE_complex_division_problem_l1339_133931

theorem complex_division_problem (z : ℂ) (h : z = 4 + 3*I) : 
  Complex.abs z / z = 4/5 - 3/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_problem_l1339_133931


namespace NUMINAMATH_CALUDE_banana_bunches_l1339_133946

theorem banana_bunches (total_bananas : ℕ) (eight_bunch_count : ℕ) (bananas_per_eight_bunch : ℕ) : 
  total_bananas = 83 →
  eight_bunch_count = 6 →
  bananas_per_eight_bunch = 8 →
  ∃ (seven_bunch_count : ℕ),
    seven_bunch_count * 7 + eight_bunch_count * bananas_per_eight_bunch = total_bananas ∧
    seven_bunch_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_banana_bunches_l1339_133946


namespace NUMINAMATH_CALUDE_enrollment_analysis_l1339_133906

def summit_ridge : ℕ := 1560
def pine_hills : ℕ := 1150
def oak_valley : ℕ := 1950
def maple_town : ℕ := 1840

def enrollments : List ℕ := [summit_ridge, pine_hills, oak_valley, maple_town]

theorem enrollment_analysis :
  (List.maximum enrollments).get! - (List.minimum enrollments).get! = 800 ∧
  (List.sum enrollments) / enrollments.length = 1625 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_analysis_l1339_133906


namespace NUMINAMATH_CALUDE_complex_number_plus_modulus_l1339_133981

theorem complex_number_plus_modulus (z : ℂ) : 
  z + Complex.abs z = 5 + Complex.I * Real.sqrt 3 → z = 11/5 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_plus_modulus_l1339_133981


namespace NUMINAMATH_CALUDE_initials_probability_l1339_133903

/-- The number of students in the class -/
def class_size : ℕ := 30

/-- The number of consonants in the alphabet (excluding Y) -/
def num_consonants : ℕ := 21

/-- The number of consonants we're interested in (B, C, D) -/
def target_consonants : ℕ := 3

/-- The probability of selecting a student with initials starting with B, C, or D -/
def probability : ℚ := 1 / 21

theorem initials_probability :
  probability = (min class_size (target_consonants * (num_consonants - 1))) / (class_size * num_consonants) :=
sorry

end NUMINAMATH_CALUDE_initials_probability_l1339_133903


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l1339_133995

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 : ℚ) + ((1 / 3 : ℚ) - (1 / 8 : ℚ)) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l1339_133995


namespace NUMINAMATH_CALUDE_clothes_fraction_is_one_eighth_l1339_133908

/-- The fraction of Gina's initial money used to buy clothes -/
def fraction_for_clothes (initial_amount : ℚ) 
  (fraction_to_mom : ℚ) (fraction_to_charity : ℚ) (amount_kept : ℚ) : ℚ :=
  let amount_to_mom := initial_amount * fraction_to_mom
  let amount_to_charity := initial_amount * fraction_to_charity
  let amount_for_clothes := initial_amount - amount_to_mom - amount_to_charity - amount_kept
  amount_for_clothes / initial_amount

theorem clothes_fraction_is_one_eighth :
  fraction_for_clothes 400 (1/4) (1/5) 170 = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_clothes_fraction_is_one_eighth_l1339_133908


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_midpoint_segment_length_l1339_133945

/-- A trapezoid with upper base length L and midline length m -/
structure Trapezoid (L m : ℝ) where
  upper_base : ℝ := L
  midline : ℝ := m

/-- The length of the segment connecting the midpoints of the two diagonals in a trapezoid -/
def diagonal_midpoint_segment_length (T : Trapezoid L m) : ℝ :=
  T.midline - T.upper_base

theorem trapezoid_diagonal_midpoint_segment_length (L m : ℝ) (T : Trapezoid L m) :
  diagonal_midpoint_segment_length T = m - L := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_midpoint_segment_length_l1339_133945


namespace NUMINAMATH_CALUDE_exponent_division_l1339_133935

theorem exponent_division (a : ℝ) : a^3 / a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1339_133935


namespace NUMINAMATH_CALUDE_determine_c_absolute_value_l1339_133963

/-- The polynomial g(x) = ax^4 + bx^3 + cx^2 + bx + a -/
def g (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

/-- The theorem statement -/
theorem determine_c_absolute_value (a b c : ℤ) : 
  g a b c (3 + Complex.I) = 0 ∧ 
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 111 := by
  sorry

end NUMINAMATH_CALUDE_determine_c_absolute_value_l1339_133963


namespace NUMINAMATH_CALUDE_ronas_age_l1339_133989

theorem ronas_age (rona rachel collete : ℕ) 
  (h1 : rachel = 2 * rona)
  (h2 : collete = rona / 2)
  (h3 : rachel - collete = 12) : 
  rona = 12 := by
sorry

end NUMINAMATH_CALUDE_ronas_age_l1339_133989


namespace NUMINAMATH_CALUDE_function_inequality_l1339_133936

theorem function_inequality (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x * Real.log x - a * x ≥ -x^2 - 2) → a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1339_133936


namespace NUMINAMATH_CALUDE_sqrt_square_eq_x_l1339_133969

theorem sqrt_square_eq_x (x : ℝ) (h : x ≥ 0) : (Real.sqrt x)^2 = x := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_x_l1339_133969


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l1339_133923

/-- The last two nonzero digits of n! -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The last two nonzero digits of 80! are 08 -/
theorem last_two_nonzero_digits_80_factorial :
  lastTwoNonzeroDigits 80 = 8 := by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l1339_133923


namespace NUMINAMATH_CALUDE_democrat_ratio_l1339_133925

theorem democrat_ratio (total_participants : ℕ) 
  (female_democrats : ℕ) (total_democrats : ℕ) :
  total_participants = 750 →
  female_democrats = 125 →
  total_democrats = total_participants / 3 →
  female_democrats * 2 ≤ total_participants →
  (total_democrats - female_democrats) * 4 = 
    total_participants - female_democrats * 2 := by
  sorry

end NUMINAMATH_CALUDE_democrat_ratio_l1339_133925


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1339_133958

theorem infinite_series_sum : 
  (∑' n : ℕ, n / (5 ^ n : ℝ)) = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1339_133958


namespace NUMINAMATH_CALUDE_sum_of_twenty_numbers_l1339_133915

theorem sum_of_twenty_numbers : 
  let numbers : List Nat := [87, 91, 94, 88, 93, 91, 89, 87, 92, 86, 90, 92, 88, 90, 91, 86, 89, 92, 95, 88]
  numbers.sum = 1799 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_twenty_numbers_l1339_133915


namespace NUMINAMATH_CALUDE_class_size_problem_l1339_133905

theorem class_size_problem (class_a class_b class_c : ℕ) : 
  class_a = 2 * class_b →
  class_a = class_c / 3 →
  class_c = 120 →
  class_b = 20 := by
sorry

end NUMINAMATH_CALUDE_class_size_problem_l1339_133905


namespace NUMINAMATH_CALUDE_pat_calculation_l1339_133948

theorem pat_calculation (x : ℝ) : (x / 8 + 2 * x - 12 = 40) → (8 * x + 2 * x + 12 > 1000) := by
  sorry

end NUMINAMATH_CALUDE_pat_calculation_l1339_133948


namespace NUMINAMATH_CALUDE_pastries_sold_l1339_133982

/-- Given that a baker initially had 56 pastries and now has 27 pastries remaining,
    prove that the number of pastries sold is 29. -/
theorem pastries_sold (initial_pastries : ℕ) (remaining_pastries : ℕ) 
  (h1 : initial_pastries = 56) (h2 : remaining_pastries = 27) : 
  initial_pastries - remaining_pastries = 29 := by
  sorry

end NUMINAMATH_CALUDE_pastries_sold_l1339_133982


namespace NUMINAMATH_CALUDE_smallest_term_at_four_l1339_133966

def a (n : ℕ+) : ℚ := (1 / 3) * n^3 - 13 * n

theorem smallest_term_at_four :
  ∀ k : ℕ+, a 4 ≤ a k := by sorry

end NUMINAMATH_CALUDE_smallest_term_at_four_l1339_133966


namespace NUMINAMATH_CALUDE_cost_of_stationery_l1339_133927

/-- Given the cost of different combinations of erasers, pens, and markers,
    prove that 3 erasers, 4 pens, and 6 markers cost 520 rubles. -/
theorem cost_of_stationery (E P M : ℕ) : 
  (E + 3 * P + 2 * M = 240) →
  (2 * E + 5 * P + 4 * M = 440) →
  (3 * E + 4 * P + 6 * M = 520) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_stationery_l1339_133927


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_75_4410_l1339_133977

theorem gcd_lcm_sum_75_4410 : Nat.gcd 75 4410 + Nat.lcm 75 4410 = 22065 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_75_4410_l1339_133977


namespace NUMINAMATH_CALUDE_f_extrema_on_3_to_5_f_extrema_on_neg1_to_3_l1339_133940

-- Define the function f
def f (x : ℝ) := x^2 - 4*x + 3

-- Theorem for the first interval [3, 5]
theorem f_extrema_on_3_to_5 :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 3 5, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 3 5, f x = max) ∧
    (∀ x ∈ Set.Icc 3 5, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 3 5, f x = min) ∧
    max = 8 ∧ min = 0 :=
sorry

-- Theorem for the second interval [-1, 3]
theorem f_extrema_on_neg1_to_3 :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-1) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = min) ∧
    max = 8 ∧ min = -1 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_on_3_to_5_f_extrema_on_neg1_to_3_l1339_133940


namespace NUMINAMATH_CALUDE_divisibility_of_A_l1339_133984

def A : ℕ := 2013 * (10^(4*165) - 1) / (10^4 - 1)

theorem divisibility_of_A : 2013^2 ∣ A := by sorry

end NUMINAMATH_CALUDE_divisibility_of_A_l1339_133984


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_l1339_133954

/-- For all positive integers n > 2, the greatest common divisor of n^5 - 5n^3 + 4n is 120. -/
theorem gcd_of_polynomial (n : ℕ) (h : n > 2) : Nat.gcd (n^5 - 5*n^3 + 4*n) 120 = 120 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_l1339_133954


namespace NUMINAMATH_CALUDE_teachers_day_theorem_l1339_133904

/-- A directed graph with 200 vertices where each vertex has exactly one outgoing edge -/
structure TeacherGraph where
  vertices : Finset (Fin 200)
  edges : Fin 200 → Fin 200
  edge_property : ∀ v, v ∈ vertices → edges v ≠ v

/-- An independent set in the graph -/
def IndependentSet (G : TeacherGraph) (S : Finset (Fin 200)) : Prop :=
  ∀ u v, u ∈ S → v ∈ S → u ≠ v → G.edges u ≠ v

/-- The theorem stating that there exists an independent set of size at least 67 -/
theorem teachers_day_theorem (G : TeacherGraph) :
  ∃ S : Finset (Fin 200), IndependentSet G S ∧ S.card ≥ 67 := by
  sorry


end NUMINAMATH_CALUDE_teachers_day_theorem_l1339_133904


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_range_part2_l1339_133907

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 2 ≥ 3} = {x : ℝ | x ≤ -3/4 ∨ x ≥ 3/4} := by sorry

-- Part 2
theorem solution_range_part2 :
  ∀ a : ℝ, a > 0 → (∃ x : ℝ, f x a < a/2 + 1) ↔ a > 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_range_part2_l1339_133907


namespace NUMINAMATH_CALUDE_percentage_calculation_l1339_133912

theorem percentage_calculation (n : ℝ) (h : n = 6000) : (0.1 * (0.3 * (0.5 * n))) = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1339_133912


namespace NUMINAMATH_CALUDE_min_product_of_prime_sum_l1339_133987

theorem min_product_of_prime_sum (m n p : ℕ) : 
  Prime m → Prime n → Prime p → 
  m ≠ n → m ≠ p → n ≠ p → 
  m + n = p → 
  (∀ m' n' p' : ℕ, Prime m' → Prime n' → Prime p' → 
    m' ≠ n' → m' ≠ p' → n' ≠ p' → 
    m' + n' = p' → m' * n' * p' ≥ m * n * p) → 
  m * n * p = 30 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_prime_sum_l1339_133987


namespace NUMINAMATH_CALUDE_items_in_bags_distribution_l1339_133974

theorem items_in_bags_distribution (n : Nat) (k : Nat) : 
  n = 6 → k = 3 → (
    (1 : Nat) + -- All items in one bag
    n + -- n-1 items in one bag, 1 in another
    (n.choose 4) + -- 4 items in one bag, 2 in another
    (n.choose 4) + -- 4 items in one bag, 1 each in the other two
    (n.choose 3 / 2) + -- 3 items in each of two bags
    (n.choose 2 * (n - 2).choose 2 / 6) -- 2 items in each bag
  ) = 62 := by
  sorry

end NUMINAMATH_CALUDE_items_in_bags_distribution_l1339_133974


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l1339_133951

theorem quadratic_function_k_value (a b c k : ℤ) :
  let g := fun (x : ℤ) => a * x^2 + b * x + c
  g 1 = 0 ∧
  20 < g 5 ∧ g 5 < 30 ∧
  40 < g 6 ∧ g 6 < 50 ∧
  3000 * k < g 100 ∧ g 100 < 3000 * (k + 1) →
  k = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l1339_133951


namespace NUMINAMATH_CALUDE_partner_a_profit_share_l1339_133914

/-- Calculates the share of profit for partner A given the initial investments,
    changes after 8 months, and total profit at the end of the year. -/
theorem partner_a_profit_share
  (a_initial : ℕ)
  (b_initial : ℕ)
  (a_change : ℤ)
  (b_change : ℕ)
  (total_profit : ℕ)
  (h1 : a_initial = 6000)
  (h2 : b_initial = 4000)
  (h3 : a_change = -1000)
  (h4 : b_change = 1000)
  (h5 : total_profit = 630) :
  ((a_initial * 8 + (a_initial + a_change) * 4) * total_profit) /
  ((a_initial * 8 + (a_initial + a_change) * 4) + (b_initial * 8 + (b_initial + b_change) * 4)) = 357 :=
by sorry

end NUMINAMATH_CALUDE_partner_a_profit_share_l1339_133914


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_plus_1000_l1339_133934

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_plus_1000 :
  units_digit (sum_factorials 10 + 1000) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_plus_1000_l1339_133934


namespace NUMINAMATH_CALUDE_basketball_cricket_students_l1339_133992

theorem basketball_cricket_students (basketball : ℕ) (cricket : ℕ) (both : ℕ)
  (h1 : basketball = 12)
  (h2 : cricket = 8)
  (h3 : both = 3) :
  basketball + cricket - both = 17 := by
sorry

end NUMINAMATH_CALUDE_basketball_cricket_students_l1339_133992


namespace NUMINAMATH_CALUDE_chord_length_l1339_133953

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (x y : ℝ) : 
  (2 * x - y - 1 = 0) → 
  ((x - 2)^2 + (y + 2)^2 = 9) → 
  ∃ (p q : ℝ × ℝ), 
    (2 * p.1 - p.2 - 1 = 0) ∧ 
    ((p.1 - 2)^2 + (p.2 + 2)^2 = 9) ∧ 
    (2 * q.1 - q.2 - 1 = 0) ∧ 
    ((q.1 - 2)^2 + (q.2 + 2)^2 = 9) ∧ 
    (p ≠ q) ∧
    ((p.1 - q.1)^2 + (p.2 - q.2)^2 = 4^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l1339_133953


namespace NUMINAMATH_CALUDE_half_angle_quadrants_l1339_133978

/-- An angle is in the 4th quadrant if it's between 3π/2 and 2π (exclusive) -/
def in_fourth_quadrant (α : ℝ) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

/-- An angle is in the 2nd quadrant if it's between π/2 and π (exclusive) -/
def in_second_quadrant (α : ℝ) : Prop :=
  Real.pi / 2 < α ∧ α < Real.pi

/-- An angle is in the 4th quadrant if it's between 3π/2 and 2π (exclusive) -/
def in_fourth_quadrant_half (α : ℝ) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

theorem half_angle_quadrants (α : ℝ) :
  in_fourth_quadrant α →
  (in_second_quadrant (α/2) ∨ in_fourth_quadrant_half (α/2)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrants_l1339_133978


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_negative_five_l1339_133957

theorem sum_reciprocals_equals_negative_five (x y : ℝ) 
  (eq1 : x^2 + Real.sqrt 3 * y = 4)
  (eq2 : y^2 + Real.sqrt 3 * x = 4)
  (neq : x ≠ y) :
  y / x + x / y = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_negative_five_l1339_133957


namespace NUMINAMATH_CALUDE_abc_values_l1339_133993

theorem abc_values (a b c : ℝ) 
  (h1 : a / 2 = b / 3)
  (h2 : b / 3 = c / 4)
  (h3 : a / 2 ≠ 0)
  (h4 : 2 * a - b + c = 10) :
  a = 4 ∧ b = 6 ∧ c = 8 := by
sorry

end NUMINAMATH_CALUDE_abc_values_l1339_133993


namespace NUMINAMATH_CALUDE_conference_handshakes_theorem_l1339_133919

/-- The number of handshakes in a conference with special conditions -/
def conference_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) - (k.choose 2)

/-- Theorem: In a conference of 30 people, where 3 specific people don't shake hands with each other,
    the total number of handshakes is 432 -/
theorem conference_handshakes_theorem :
  conference_handshakes 30 3 = 432 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_theorem_l1339_133919


namespace NUMINAMATH_CALUDE_files_deleted_l1339_133962

theorem files_deleted (initial_music : ℕ) (initial_video : ℕ) (remaining : ℕ) : 
  initial_music = 26 → initial_video = 36 → remaining = 14 →
  initial_music + initial_video - remaining = 48 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l1339_133962


namespace NUMINAMATH_CALUDE_system_solution_l1339_133967

theorem system_solution (x y z u v : ℝ) : 
  (x + y + z + u = 5) ∧
  (y + z + u + v = 1) ∧
  (z + u + v + x = 2) ∧
  (u + v + x + y = 0) ∧
  (v + x + y + z = 4) →
  (v = -2 ∧ x = 2 ∧ y = 1 ∧ z = 3 ∧ u = -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1339_133967


namespace NUMINAMATH_CALUDE_sum_coordinates_endpoint_l1339_133947

/-- Given a line segment CD with midpoint M(4,7) and endpoint C(6,2),
    the sum of coordinates of the other endpoint D is 14. -/
theorem sum_coordinates_endpoint (C D M : ℝ × ℝ) : 
  C = (6, 2) → M = (4, 7) → M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_endpoint_l1339_133947


namespace NUMINAMATH_CALUDE_negative_pi_less_than_negative_three_l1339_133921

theorem negative_pi_less_than_negative_three : -Real.pi < -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_pi_less_than_negative_three_l1339_133921


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l1339_133970

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the closed interval [1, 5]
def I : Set ℝ := Set.Icc 1 5

-- Theorem statement
theorem max_min_values_of_f :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 11 ∧ f b = 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l1339_133970


namespace NUMINAMATH_CALUDE_parallelogram_rotation_volume_ratio_l1339_133971

/-- Given a parallelogram with adjacent sides a and b, the ratio of the volume of the cylinder
    formed by rotating the parallelogram around side a to the volume of the cylinder formed by
    rotating the parallelogram around side b is equal to a/b. -/
theorem parallelogram_rotation_volume_ratio
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (π * (a/2)^2 * b) / (π * (b/2)^2 * a) = a / b :=
sorry

end NUMINAMATH_CALUDE_parallelogram_rotation_volume_ratio_l1339_133971


namespace NUMINAMATH_CALUDE_expression_simplification_l1339_133939

theorem expression_simplification (m n : ℤ) (hm : m = 1) (hn : n = -2) :
  3 * m^2 * n + 2 * (2 * m * n^2 - 3 * m^2 * n) - 3 * (m * n^2 - m^2 * n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1339_133939


namespace NUMINAMATH_CALUDE_power_of_four_exponent_l1339_133910

theorem power_of_four_exponent (n : ℕ) (x : ℕ) 
  (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (hn : n = 21) : x = 22 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_exponent_l1339_133910


namespace NUMINAMATH_CALUDE_mother_daughter_age_difference_l1339_133968

theorem mother_daughter_age_difference :
  ∀ (mother_age daughter_age : ℕ),
    mother_age = 55 →
    mother_age - 1 = 2 * (daughter_age - 1) →
    mother_age - daughter_age = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_mother_daughter_age_difference_l1339_133968


namespace NUMINAMATH_CALUDE_distance_between_points_l1339_133960

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, -3)
  let p2 : ℝ × ℝ := (5, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1339_133960


namespace NUMINAMATH_CALUDE_power_difference_value_l1339_133916

theorem power_difference_value (a x y : ℝ) (ha : a > 0) (hx : a^x = 2) (hy : a^y = 3) :
  a^(x - y) = 2/3 := by sorry

end NUMINAMATH_CALUDE_power_difference_value_l1339_133916
