import Mathlib

namespace NUMINAMATH_GPT_mod_remainder_l109_10925

theorem mod_remainder (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end NUMINAMATH_GPT_mod_remainder_l109_10925


namespace NUMINAMATH_GPT_complex_mul_im_unit_l109_10986

theorem complex_mul_im_unit (i : ℂ) (h : i^2 = -1) : i * (1 - i) = 1 + i := by
  sorry

end NUMINAMATH_GPT_complex_mul_im_unit_l109_10986


namespace NUMINAMATH_GPT_proposition_verification_l109_10991

-- Definitions and Propositions
def prop1 : Prop := (∀ x, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x, x ≠ 1 ∧ x^2 - 3 * x + 2 = 0)
def prop2 : Prop := (∀ x, ¬ (x^2 - 3 * x + 2 = 0 → x = 1) → (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0))
def prop3 : Prop := ¬ (∃ x > 0, x^2 + x + 1 < 0) → (∀ x ≤ 0, x^2 + x + 1 ≥ 0)
def prop4 : Prop := ¬ (∃ p q : Prop, (p ∨ q) → ¬p ∧ ¬q)

-- Final theorem statement
theorem proposition_verification : prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 := by 
  sorry

end NUMINAMATH_GPT_proposition_verification_l109_10991


namespace NUMINAMATH_GPT_place_two_after_three_digit_number_l109_10983

theorem place_two_after_three_digit_number (h t u : ℕ) 
  (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) : 
  (100 * h + 10 * t + u) * 10 + 2 = 1000 * h + 100 * t + 10 * u + 2 := 
by
  sorry

end NUMINAMATH_GPT_place_two_after_three_digit_number_l109_10983


namespace NUMINAMATH_GPT_tangent_line_b_value_l109_10918

theorem tangent_line_b_value (b : ℝ) : 
  (∃ pt : ℝ × ℝ, (pt.1)^2 + (pt.2)^2 = 25 ∧ pt.1 - pt.2 + b = 0)
  ↔ b = 5 * Real.sqrt 2 ∨ b = -5 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_b_value_l109_10918


namespace NUMINAMATH_GPT_velma_more_than_veronica_l109_10931

-- Defining the distances each flashlight can be seen
def veronica_distance : ℕ := 1000
def freddie_distance : ℕ := 3 * veronica_distance
def velma_distance : ℕ := 5 * freddie_distance - 2000

-- The proof problem: Prove that Velma's flashlight can be seen 12000 feet farther than Veronica's flashlight.
theorem velma_more_than_veronica : velma_distance - veronica_distance = 12000 := by
  sorry

end NUMINAMATH_GPT_velma_more_than_veronica_l109_10931


namespace NUMINAMATH_GPT_determine_m_l109_10907

theorem determine_m (a b c m : ℤ) 
  (h1 : c = -4 * a - 2 * b)
  (h2 : 70 < 4 * (8 * a + b) ∧ 4 * (8 * a + b) < 80)
  (h3 : 110 < 5 * (9 * a + b) ∧ 5 * (9 * a + b) < 120)
  (h4 : 2000 * m < (2500 * a + 50 * b + c) ∧ (2500 * a + 50 * b + c) < 2000 * (m + 1)) :
  m = 5 := sorry

end NUMINAMATH_GPT_determine_m_l109_10907


namespace NUMINAMATH_GPT_opposite_of_neg_two_is_two_l109_10957

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_two_is_two_l109_10957


namespace NUMINAMATH_GPT_distinct_L_shapes_l109_10956

-- Definitions of conditions
def num_convex_shapes : Nat := 10
def L_shapes_per_convex : Nat := 2
def corner_L_shapes : Nat := 4

-- Total number of distinct "L" shapes
def total_L_shapes : Nat :=
  num_convex_shapes * L_shapes_per_convex + corner_L_shapes

theorem distinct_L_shapes :
  total_L_shapes = 24 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_distinct_L_shapes_l109_10956


namespace NUMINAMATH_GPT_H_double_prime_coordinates_l109_10947

/-- Define the points of the parallelogram EFGH and their reflections. --/
structure Point := (x : ℝ) (y : ℝ)

def E : Point := ⟨3, 4⟩
def F : Point := ⟨5, 7⟩
def G : Point := ⟨7, 4⟩
def H : Point := ⟨5, 1⟩

/-- Reflection of a point across the x-axis changes the y-coordinate sign. --/
def reflect_x (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Reflection of a point across y=x-1 involves translation and reflection across y=x. --/
def reflect_y_x_minus_1 (p : Point) : Point :=
  let translated := Point.mk p.x (p.y + 1)
  let reflected := Point.mk translated.y translated.x
  Point.mk reflected.x (reflected.y - 1)

def H' : Point := reflect_x H
def H'' : Point := reflect_y_x_minus_1 H'

theorem H_double_prime_coordinates : H'' = ⟨0, 4⟩ :=
by
  sorry

end NUMINAMATH_GPT_H_double_prime_coordinates_l109_10947


namespace NUMINAMATH_GPT_gcd_polynomial_multiple_of_532_l109_10937

theorem gcd_polynomial_multiple_of_532 (a : ℤ) (h : ∃ k : ℤ, a = 532 * k) :
  Int.gcd (5 * a ^ 3 + 2 * a ^ 2 + 6 * a + 76) a = 76 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_multiple_of_532_l109_10937


namespace NUMINAMATH_GPT_matt_total_vibrations_l109_10960

noncomputable def vibrations_lowest : ℕ := 1600
noncomputable def vibrations_highest : ℕ := vibrations_lowest + (6 * vibrations_lowest / 10)
noncomputable def time_seconds : ℕ := 300
noncomputable def total_vibrations : ℕ := vibrations_highest * time_seconds

theorem matt_total_vibrations :
  total_vibrations = 768000 := by
  sorry

end NUMINAMATH_GPT_matt_total_vibrations_l109_10960


namespace NUMINAMATH_GPT_white_area_of_painting_l109_10974

theorem white_area_of_painting (s : ℝ) (total_gray_area : ℝ) (gray_area_squares : ℕ)
  (h1 : ∀ t, t = 3 * s) -- The frame is 3 times the smaller square's side length.
  (h2 : total_gray_area = 62) -- The gray area is 62 cm^2.
  (h3 : gray_area_squares = 31) -- The gray area is composed of 31 smaller squares.
  : ∃ white_area, white_area = 10 := 
  sorry

end NUMINAMATH_GPT_white_area_of_painting_l109_10974


namespace NUMINAMATH_GPT_min_value_x_2y_l109_10954

theorem min_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y + 2 * x * y = 8) : x + 2 * y ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_x_2y_l109_10954


namespace NUMINAMATH_GPT_sales_value_minimum_l109_10989

theorem sales_value_minimum (V : ℝ) (base_salary new_salary : ℝ) (commission_rate sales_needed old_salary : ℝ)
    (h_base_salary : base_salary = 45000 )
    (h_new_salary : new_salary = base_salary + 0.15 * V * sales_needed)
    (h_sales_needed : sales_needed = 266.67)
    (h_old_salary : old_salary = 75000) :
    new_salary ≥ old_salary ↔ V ≥ 750 := 
by
  sorry

end NUMINAMATH_GPT_sales_value_minimum_l109_10989


namespace NUMINAMATH_GPT_dilation_image_l109_10953

open Complex

noncomputable def dilation_center := (1 : ℂ) + (3 : ℂ) * I
noncomputable def scale_factor := -3
noncomputable def initial_point := -I
noncomputable def target_point := (4 : ℂ) + (15 : ℂ) * I

theorem dilation_image :
  let c := dilation_center
  let k := scale_factor
  let z := initial_point
  let z_prime := target_point
  z_prime = c + k * (z - c) := 
  by
    sorry

end NUMINAMATH_GPT_dilation_image_l109_10953


namespace NUMINAMATH_GPT_cars_meet_cars_apart_l109_10952

section CarsProblem

variable (distance : ℕ) (speedA speedB : ℕ) (distanceToMeet distanceApart : ℕ)

def meetTime := distance / (speedA + speedB)
def apartTime1 := (distance - distanceApart) / (speedA + speedB)
def apartTime2 := (distance + distanceApart) / (speedA + speedB)

theorem cars_meet (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85):
  meetTime distance speedA speedB = 9 / 4 := by
  sorry

theorem cars_apart (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85) (h4: distanceApart = 50):
  apartTime1 distance speedA speedB distanceApart = 2 ∧ apartTime2 distance speedA speedB distanceApart = 5 / 2 := by
  sorry

end CarsProblem

end NUMINAMATH_GPT_cars_meet_cars_apart_l109_10952


namespace NUMINAMATH_GPT_seq_positive_integers_no_m_exists_l109_10943

-- Definition of the sequence
def seq (n : ℕ) : ℕ :=
  Nat.recOn n
    1
    (λ n a_n => 3 * a_n + 2 * (2 * a_n * a_n - 1).sqrt)

-- Axiomatize the properties involved in the recurrence relation
axiom rec_sqrt_property (n : ℕ) : ∃ k : ℕ, (2 * seq n * seq n - 1) = k * k

-- Proof statement for the sequence of positive integers
theorem seq_positive_integers (n : ℕ) : seq n > 0 := sorry

-- Proof statement for non-existence of m such that 2015 divides seq(m)
theorem no_m_exists (m : ℕ) : ¬ (2015 ∣ seq m) := sorry

end NUMINAMATH_GPT_seq_positive_integers_no_m_exists_l109_10943


namespace NUMINAMATH_GPT_x_minus_y_eq_2_l109_10940

theorem x_minus_y_eq_2 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : 3 * x + 2 * y = 11) : x - y = 2 :=
sorry

end NUMINAMATH_GPT_x_minus_y_eq_2_l109_10940


namespace NUMINAMATH_GPT_non_raining_hours_l109_10916

-- Definitions based on the conditions.
def total_hours := 9
def rained_hours := 4

-- Problem statement: Prove that the non-raining hours equals to 5 given total_hours and rained_hours.
theorem non_raining_hours : (total_hours - rained_hours = 5) :=
by
  -- The proof is omitted with "sorry" to indicate the missing proof.
  sorry

end NUMINAMATH_GPT_non_raining_hours_l109_10916


namespace NUMINAMATH_GPT_spending_difference_l109_10913

def chocolate_price : ℝ := 7
def candy_bar_price : ℝ := 2
def discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def gum_price : ℝ := 3

def discounted_chocolate_price : ℝ := chocolate_price * (1 - discount_rate)
def total_before_tax : ℝ := candy_bar_price + gum_price
def tax_amount : ℝ := total_before_tax * sales_tax_rate
def total_after_tax : ℝ := total_before_tax + tax_amount

theorem spending_difference : 
  discounted_chocolate_price - candy_bar_price = 3.95 :=
by 
  -- Apply the necessary calculations
  have discount_chocolate : ℝ := discounted_chocolate_price
  have candy_bar : ℝ := candy_bar_price
  calc
    discounted_chocolate_price - candy_bar_price = _ := sorry

end NUMINAMATH_GPT_spending_difference_l109_10913


namespace NUMINAMATH_GPT_vector_parallel_dot_product_l109_10912

theorem vector_parallel_dot_product (x : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (x, 1))
  (h2 : b = (4, 2))
  (h3 : x / 4 = 1 / 2) : 
  (a.1 * (b.1 - a.1) + a.2 * (b.2 - a.2)) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_vector_parallel_dot_product_l109_10912


namespace NUMINAMATH_GPT_solve_for_x_l109_10987

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 15 / (x / 3)) : x = 15 * Real.sqrt 3 ∨ x = -15 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l109_10987


namespace NUMINAMATH_GPT_domain_translation_l109_10945

theorem domain_translation (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < 3 * x + 2 ∧ 3 * x + 2 < 1 → (∃ y : ℝ, f (3 * x + 2) = y)) →
  (∀ x : ℝ, ∃ y : ℝ, f (2 * x - 1) = y ↔ (3 / 2) < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_domain_translation_l109_10945


namespace NUMINAMATH_GPT_blood_drug_concentration_at_13_hours_l109_10975

theorem blood_drug_concentration_at_13_hours :
  let peak_time := 3
  let test_interval := 2
  let decrease_rate := 0.4
  let target_rate := 0.01024
  let time_to_reach_target := (fun n => (2 * n + 1))
  peak_time + test_interval * 5 = 13 :=
sorry

end NUMINAMATH_GPT_blood_drug_concentration_at_13_hours_l109_10975


namespace NUMINAMATH_GPT_seq_general_formula_l109_10988

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a n ^ 2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0

theorem seq_general_formula {a : ℕ → ℝ} (h1 : a 1 = 1) (h2 : seq a) :
  ∀ n, a n = 1 / 2 ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_seq_general_formula_l109_10988


namespace NUMINAMATH_GPT_find_x_l109_10971

theorem find_x (x : ℕ) (h : 2^x - 2^(x-2) = 3 * 2^(12)) : x = 14 :=
sorry

end NUMINAMATH_GPT_find_x_l109_10971


namespace NUMINAMATH_GPT_average_speed_l109_10997

def initial_odometer_reading : ℕ := 20
def final_odometer_reading : ℕ := 200
def travel_duration : ℕ := 6

theorem average_speed :
  (final_odometer_reading - initial_odometer_reading) / travel_duration = 30 := by
  sorry

end NUMINAMATH_GPT_average_speed_l109_10997


namespace NUMINAMATH_GPT_digital_earth_functionalities_l109_10920

def digital_earth_allows_internet_navigation : Prop := 
  ∀ (f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"]

def digital_earth_does_not_allow_physical_travel : Prop := 
  ¬ (∀ (f : String), f ∈ ["Travel around the world"])

theorem digital_earth_functionalities :
  digital_earth_allows_internet_navigation ∧ digital_earth_does_not_allow_physical_travel →
  ∀(f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"] :=
by
  sorry

end NUMINAMATH_GPT_digital_earth_functionalities_l109_10920


namespace NUMINAMATH_GPT_count_implications_l109_10958

def r : Prop := sorry
def s : Prop := sorry

def statement_1 := ¬r ∧ ¬s
def statement_2 := ¬r ∧ s
def statement_3 := r ∧ ¬s
def statement_4 := r ∧ s

def neg_rs : Prop := r ∨ s

theorem count_implications : (statement_2 → neg_rs) ∧ 
                             (statement_3 → neg_rs) ∧ 
                             (statement_4 → neg_rs) ∧ 
                             (¬(statement_1 → neg_rs)) -> 
                             3 = 3 := by
  sorry

end NUMINAMATH_GPT_count_implications_l109_10958


namespace NUMINAMATH_GPT_digital_earth_storage_technology_matured_l109_10972

-- Definitions of conditions as technology properties
def NanoStorageTechnology : Prop := 
  -- Assume it has matured (based on solution analysis)
  sorry

def LaserHolographicStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def ProteinStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def DistributedStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def VirtualStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

def SpatialStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

def VisualizationStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

-- Lean statement to prove the combination
theorem digital_earth_storage_technology_matured : 
  NanoStorageTechnology ∧ LaserHolographicStorageTechnology ∧ ProteinStorageTechnology ∧ DistributedStorageTechnology :=
by {
  sorry
}

end NUMINAMATH_GPT_digital_earth_storage_technology_matured_l109_10972


namespace NUMINAMATH_GPT_ada_original_seat_l109_10982

theorem ada_original_seat (seats: Fin 6 → Option String)
  (Bea_init Ceci_init Dee_init Edie_init Fran_init: Fin 6) 
  (Bea_fin Ceci_fin Fran_fin: Fin 6) 
  (Ada_fin: Fin 6)
  (Bea_moves_right: Bea_fin = Bea_init + 3)
  (Ceci_stays: Ceci_fin = Ceci_init)
  (Dee_switches_with_Edie: ∃ Dee_fin Edie_fin: Fin 6, Dee_fin = Edie_init ∧ Edie_fin = Dee_init)
  (Fran_moves_left: Fran_fin = Fran_init - 1)
  (Ada_end_seat: Ada_fin = 0 ∨ Ada_fin = 5):
  ∃ Ada_init: Fin 6, Ada_init = 2 + Ada_fin + 1 → Ada_init = 3 := 
by 
  sorry

end NUMINAMATH_GPT_ada_original_seat_l109_10982


namespace NUMINAMATH_GPT_alpha_values_m_range_l109_10994

noncomputable section

open Real

def f (x : ℝ) (α : ℝ) : ℝ := 2^(x + cos α) - 2^(-x + cos α)

-- Problem 1: Set of values for α
theorem alpha_values (h : f 1 α = 3/4) : ∃ k : ℤ, α = 2 * k * π + π :=
sorry

-- Problem 2: Range of values for real number m
theorem m_range (h0 : 0 ≤ θ ∧ θ ≤ π / 2) 
  (h1 : ∀ (m : ℝ), f (m * cos θ) (-1) + f (1 - m) (-1) > 0) : 
  ∀ (m : ℝ), m < 1 :=
sorry

end NUMINAMATH_GPT_alpha_values_m_range_l109_10994


namespace NUMINAMATH_GPT_total_distance_proof_l109_10938

-- Define the conditions
def first_half_time := 20
def second_half_time := 30
def average_time_per_kilometer := 5

-- Calculate the total time
def total_time := first_half_time + second_half_time

-- State the proof problem: prove that the total distance is 10 kilometers
theorem total_distance_proof : 
  (total_time / average_time_per_kilometer) = 10 :=
  by sorry

end NUMINAMATH_GPT_total_distance_proof_l109_10938


namespace NUMINAMATH_GPT_solve_quadratic_l109_10970

theorem solve_quadratic : 
  ∀ x : ℝ, (x - 1) ^ 2 = 64 → (x = 9 ∨ x = -7) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l109_10970


namespace NUMINAMATH_GPT_find_a_value_l109_10962

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end NUMINAMATH_GPT_find_a_value_l109_10962


namespace NUMINAMATH_GPT_count_pairs_divisible_by_nine_l109_10914

open Nat

theorem count_pairs_divisible_by_nine (n : ℕ) (h : n = 528) :
  ∃ (count : ℕ), count = n ∧
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 100 ∧ (a^2 + b^2 + a * b) % 9 = 0 ↔
  count = 528 :=
by
  sorry

end NUMINAMATH_GPT_count_pairs_divisible_by_nine_l109_10914


namespace NUMINAMATH_GPT_find_f3_l109_10985

theorem find_f3 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_f3_l109_10985


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l109_10961

theorem trigonometric_identity_proof :
  ( (Real.cos (40 * Real.pi / 180) + Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)))
  / (Real.sin (70 * Real.pi / 180) * Real.sqrt (1 + Real.cos (40 * Real.pi / 180))) ) =
  Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l109_10961


namespace NUMINAMATH_GPT_cos_double_angle_l109_10905

open Real

-- Define the given conditions
variables {θ : ℝ}
axiom θ_in_interval : 0 < θ ∧ θ < π / 2
axiom sin_minus_cos : sin θ - cos θ = sqrt 2 / 2

-- Create a theorem that reflects the proof problem
theorem cos_double_angle : cos (2 * θ) = - sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l109_10905


namespace NUMINAMATH_GPT_student_score_variance_l109_10900

noncomputable def variance_student_score : ℝ :=
  let number_of_questions := 25
  let probability_correct := 0.8
  let score_correct := 4
  let variance_eta := number_of_questions * probability_correct * (1 - probability_correct)
  let variance_xi := (score_correct ^ 2) * variance_eta
  variance_xi

theorem student_score_variance : variance_student_score = 64 := by
  sorry

end NUMINAMATH_GPT_student_score_variance_l109_10900


namespace NUMINAMATH_GPT_minimum_time_needed_l109_10908

-- Define the task times
def review_time : ℕ := 30
def rest_time : ℕ := 30
def boil_water_time : ℕ := 15
def homework_time : ℕ := 25

-- Define the minimum time required (Xiao Ming can boil water while resting)
theorem minimum_time_needed : review_time + rest_time + homework_time = 85 := by
  -- The proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_minimum_time_needed_l109_10908


namespace NUMINAMATH_GPT_cube_divisibility_l109_10935

theorem cube_divisibility (a : ℤ) (k : ℤ) (h₁ : a > 1) 
(h₂ : (a - 1)^3 + a^3 + (a + 1)^3 = k^3) : 4 ∣ a := 
by
  sorry

end NUMINAMATH_GPT_cube_divisibility_l109_10935


namespace NUMINAMATH_GPT_inequality_solution_set_l109_10968

theorem inequality_solution_set (x : ℝ) : (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l109_10968


namespace NUMINAMATH_GPT_eggs_total_l109_10984

-- Definitions of the conditions in Lean
def num_people : ℕ := 3
def omelets_per_person : ℕ := 3
def eggs_per_omelet : ℕ := 4

-- The claim we need to prove
theorem eggs_total : (num_people * omelets_per_person) * eggs_per_omelet = 36 :=
by
  sorry

end NUMINAMATH_GPT_eggs_total_l109_10984


namespace NUMINAMATH_GPT_sum_of_v_values_is_zero_l109_10915

def v (x : ℝ) : ℝ := sorry

theorem sum_of_v_values_is_zero
  (h_odd : ∀ x : ℝ, v (-x) = -v x) :
  v (-3.14) + v (-1.57) + v (1.57) + v (3.14) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_v_values_is_zero_l109_10915


namespace NUMINAMATH_GPT_smallest_total_books_l109_10921

-- Definitions based on conditions
def physics_books (x : ℕ) := 3 * x
def chemistry_books (x : ℕ) := 2 * x
def biology_books (x : ℕ) := (3 / 2 : ℚ) * x

-- Total number of books
def total_books (x : ℕ) := physics_books x + chemistry_books x + biology_books x

-- Statement of the theorem
theorem smallest_total_books :
  ∃ x : ℕ, total_books x = 15 ∧ 
           (∀ y : ℕ, y < x → total_books y % 1 ≠ 0) :=
sorry

end NUMINAMATH_GPT_smallest_total_books_l109_10921


namespace NUMINAMATH_GPT_teaching_arrangements_l109_10979

theorem teaching_arrangements : 
  let teachers := ["A", "B", "C", "D", "E", "F"]
  let lessons := ["L1", "L2", "L3", "L4"]
  let valid_first_lesson := ["A", "B"]
  let valid_fourth_lesson := ["A", "C"]
  ∃ arrangements : ℕ, 
    (arrangements = 36) ∧
    (∀ (l1 l2 l3 l4 : String), (l1 ∈ valid_first_lesson) → (l4 ∈ valid_fourth_lesson) → 
      (l2 ≠ l1 ∧ l2 ≠ l4 ∧ l3 ≠ l1 ∧ l3 ≠ l4) ∧ 
      (List.length teachers - (if (l1 == "A") then 1 else 0) - (if (l4 == "A") then 1 else 0) = 4)) :=
by {
  -- This is just the theorem statement; no proof is required.
  sorry
}

end NUMINAMATH_GPT_teaching_arrangements_l109_10979


namespace NUMINAMATH_GPT_Emily_total_points_l109_10922

-- Definitions of the points scored in each round
def round1_points := 16
def round2_points := 32
def round3_points := -27
def round4_points := 92
def round5_points := 4

-- Total points calculation in Lean
def total_points := round1_points + round2_points + round3_points + round4_points + round5_points

-- Lean statement to prove total points at the end of the game
theorem Emily_total_points : total_points = 117 :=
by 
  -- Unfold the definition of total_points and simplify
  unfold total_points round1_points round2_points round3_points round4_points round5_points
  -- Simplify the expression
  sorry

end NUMINAMATH_GPT_Emily_total_points_l109_10922


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l109_10959

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 3 = 2 * S 2 + 1) (h2 : a 4 = 2 * S 3 + 1) :
  ∃ q : ℝ, (q = 3) :=
by
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l109_10959


namespace NUMINAMATH_GPT_amount_given_to_second_set_of_families_l109_10955

theorem amount_given_to_second_set_of_families
  (total_spent : ℝ) (amount_first_set : ℝ) (amount_last_set : ℝ)
  (h_total_spent : total_spent = 900)
  (h_amount_first_set : amount_first_set = 325)
  (h_amount_last_set : amount_last_set = 315) :
  total_spent - amount_first_set - amount_last_set = 260 :=
by
  -- sorry is placed to skip the proof
  sorry

end NUMINAMATH_GPT_amount_given_to_second_set_of_families_l109_10955


namespace NUMINAMATH_GPT_haruto_ratio_is_1_to_2_l109_10999

def haruto_tomatoes_ratio (total_tomatoes : ℕ) (eaten_by_birds : ℕ) (remaining_tomatoes : ℕ) : ℚ :=
  let picked_tomatoes := total_tomatoes - eaten_by_birds
  let given_to_friend := picked_tomatoes - remaining_tomatoes
  given_to_friend / picked_tomatoes

theorem haruto_ratio_is_1_to_2 : haruto_tomatoes_ratio 127 19 54 = 1 / 2 :=
by
  -- We'll skip the proof details as instructed
  sorry

end NUMINAMATH_GPT_haruto_ratio_is_1_to_2_l109_10999


namespace NUMINAMATH_GPT_find_wrong_quotient_l109_10933

-- Define the conditions
def correct_divisor : Nat := 21
def correct_quotient : Nat := 24
def mistaken_divisor : Nat := 12
def dividend : Nat := correct_divisor * correct_quotient

-- State the theorem to prove the wrong quotient
theorem find_wrong_quotient : (dividend / mistaken_divisor) = 42 := by
  sorry

end NUMINAMATH_GPT_find_wrong_quotient_l109_10933


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l109_10924

theorem sum_of_fourth_powers (n : ℤ) (h1 : n > 0) (h2 : (n - 1)^2 + n^2 + (n + 1)^2 = 9458) :
  (n - 1)^4 + n^4 + (n + 1)^4 = 30212622 :=
by sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l109_10924


namespace NUMINAMATH_GPT_problem_solution_l109_10904

-- Definitions of the conditions as Lean statements:
def condition1 (t : ℝ) : Prop :=
  (1 + Real.sin t) * (1 - Real.cos t) = 1

def condition2 (t : ℝ) (a b c : ℕ) : Prop :=
  (1 - Real.sin t) * (1 + Real.cos t) = (a / b) - Real.sqrt c

def areRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

-- The proof problem statement:
theorem problem_solution (t : ℝ) (a b c : ℕ) (h1 : condition1 t) (h2 : condition2 t a b c) (h3 : areRelativelyPrime a b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) : a + b + c = 2 := 
sorry

end NUMINAMATH_GPT_problem_solution_l109_10904


namespace NUMINAMATH_GPT_probability_of_exactly_three_heads_l109_10995

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_exactly_three_heads_l109_10995


namespace NUMINAMATH_GPT_babblian_word_count_l109_10936

theorem babblian_word_count (n : ℕ) (h1 : n = 6) : ∃ m, m = 258 := by
  sorry

end NUMINAMATH_GPT_babblian_word_count_l109_10936


namespace NUMINAMATH_GPT_mass_of_15_moles_is_9996_9_l109_10926

/-- Calculation of the molar mass of potassium aluminum sulfate dodecahydrate -/
def KAl_SO4_2_12H2O_molar_mass : ℝ :=
  let K := 39.10
  let Al := 26.98
  let S := 32.07
  let O := 16.00
  let H := 1.01
  K + Al + 2 * S + (8 + 24) * O + 24 * H

/-- Mass calculation for 15 moles of potassium aluminum sulfate dodecahydrate -/
def mass_of_15_moles_KAl_SO4_2_12H2O : ℝ :=
  15 * KAl_SO4_2_12H2O_molar_mass

/-- Proof statement that the mass of 15 moles of potassium aluminum sulfate dodecahydrate is 9996.9 grams -/
theorem mass_of_15_moles_is_9996_9 : mass_of_15_moles_KAl_SO4_2_12H2O = 9996.9 := by
  -- assume KAl_SO4_2_12H2O_molar_mass = 666.46 (from the problem solution steps)
  sorry

end NUMINAMATH_GPT_mass_of_15_moles_is_9996_9_l109_10926


namespace NUMINAMATH_GPT_stratified_sampling_sample_size_l109_10963

-- Definitions based on conditions
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def female_employees_in_sample : ℕ := 3

-- Proof statement
theorem stratified_sampling_sample_size : total_employees = 120 ∧ male_employees = 90 ∧ female_employees_in_sample = 3 → 
  (female_employees_in_sample + female_employees_in_sample * (male_employees / (total_employees - male_employees))) = 12 :=
sorry

end NUMINAMATH_GPT_stratified_sampling_sample_size_l109_10963


namespace NUMINAMATH_GPT_set_B_roster_method_l109_10966

def A : Set ℤ := {-2, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem set_B_roster_method : B = {4, 9, 16} :=
by
  sorry

end NUMINAMATH_GPT_set_B_roster_method_l109_10966


namespace NUMINAMATH_GPT_solve_fraction_equation_l109_10981

theorem solve_fraction_equation :
  ∀ x : ℝ, (3 / (2 * x - 2) + 1 / (1 - x) = 3) → x = 7 / 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l109_10981


namespace NUMINAMATH_GPT_Frank_seeds_per_orange_l109_10978

noncomputable def Betty_oranges := 15
noncomputable def Bill_oranges := 12
noncomputable def total_oranges := Betty_oranges + Bill_oranges
noncomputable def Frank_oranges := 3 * total_oranges
noncomputable def oranges_per_tree := 5
noncomputable def Philip_oranges := 810
noncomputable def number_of_trees := Philip_oranges / oranges_per_tree
noncomputable def seeds_per_orange := number_of_trees / Frank_oranges

theorem Frank_seeds_per_orange :
  seeds_per_orange = 2 :=
by
  sorry

end NUMINAMATH_GPT_Frank_seeds_per_orange_l109_10978


namespace NUMINAMATH_GPT_subset_single_element_l109_10969

-- Define the set X
def X : Set ℝ := { x | x > -1 }

-- The proof statement
-- We need to prove that {0} ⊆ X
theorem subset_single_element : {0} ⊆ X :=
sorry

end NUMINAMATH_GPT_subset_single_element_l109_10969


namespace NUMINAMATH_GPT_perfect_square_problem_l109_10911

-- Define the given conditions and question
theorem perfect_square_problem 
  (a b c : ℕ) 
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_cond: 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c + 1) : 
  ∃ k : ℕ, k^2 = a^2 + b^2 - a * b * c := 
sorry

end NUMINAMATH_GPT_perfect_square_problem_l109_10911


namespace NUMINAMATH_GPT_ratio_problem_l109_10910

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_problem_l109_10910


namespace NUMINAMATH_GPT_brown_gumdrops_after_replacement_l109_10948

theorem brown_gumdrops_after_replacement
  (total_gumdrops : ℕ)
  (percent_blue : ℚ)
  (percent_brown : ℚ)
  (percent_red : ℚ)
  (percent_yellow : ℚ)
  (num_green : ℕ)
  (replace_half_blue_with_brown : ℕ) :
  total_gumdrops = 120 →
  percent_blue = 0.30 →
  percent_brown = 0.20 →
  percent_red = 0.15 →
  percent_yellow = 0.10 →
  num_green = 30 →
  replace_half_blue_with_brown = 18 →
  ((percent_brown * ↑total_gumdrops) + replace_half_blue_with_brown) = 42 :=
by sorry

end NUMINAMATH_GPT_brown_gumdrops_after_replacement_l109_10948


namespace NUMINAMATH_GPT_find_larger_number_l109_10901

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 := by
  sorry

end NUMINAMATH_GPT_find_larger_number_l109_10901


namespace NUMINAMATH_GPT_gcd_max_possible_value_l109_10944

theorem gcd_max_possible_value (x y : ℤ) (h_coprime : Int.gcd x y = 1) : 
  ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
by
  sorry

end NUMINAMATH_GPT_gcd_max_possible_value_l109_10944


namespace NUMINAMATH_GPT_king_luis_courtiers_are_odd_l109_10903

theorem king_luis_courtiers_are_odd (n : ℕ) 
  (h : ∀ i : ℕ, i < n → ∃ j : ℕ, j < n ∧ i ≠ j) : 
  ¬ Even n := 
sorry

end NUMINAMATH_GPT_king_luis_courtiers_are_odd_l109_10903


namespace NUMINAMATH_GPT_original_price_of_good_l109_10973

theorem original_price_of_good (P : ℝ) (h1 : 0.684 * P = 6840) : P = 10000 :=
sorry

end NUMINAMATH_GPT_original_price_of_good_l109_10973


namespace NUMINAMATH_GPT_degrees_to_radians_150_l109_10902

theorem degrees_to_radians_150 :
  (150 : ℝ) * (Real.pi / 180) = (5 * Real.pi) / 6 :=
by
  sorry

end NUMINAMATH_GPT_degrees_to_radians_150_l109_10902


namespace NUMINAMATH_GPT_comic_books_exclusive_count_l109_10942

theorem comic_books_exclusive_count 
  (shared_comics : ℕ) 
  (total_andrew_comics : ℕ) 
  (john_exclusive_comics : ℕ) 
  (h_shared_comics : shared_comics = 15) 
  (h_total_andrew_comics : total_andrew_comics = 22) 
  (h_john_exclusive_comics : john_exclusive_comics = 10) : 
  (total_andrew_comics - shared_comics + john_exclusive_comics = 17) := by 
  sorry

end NUMINAMATH_GPT_comic_books_exclusive_count_l109_10942


namespace NUMINAMATH_GPT_find_a11_l109_10964

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom cond1 : ∀ n : ℕ, n > 0 → 4 * S n = 2 * a n - n^2 + 7 * n

-- Theorem stating the proof problem
theorem find_a11 :
  a 11 = -2 :=
sorry

end NUMINAMATH_GPT_find_a11_l109_10964


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l109_10967

noncomputable def a_n (n : ℕ) : ℝ :=
sorry

theorem arithmetic_sequence_general_formula (h1 : (a_n 2 + a_n 6) / 2 = 5)
                                            (h2 : (a_n 3 + a_n 7) / 2 = 7) :
  a_n n = 2 * (n : ℝ) - 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l109_10967


namespace NUMINAMATH_GPT_rectangle_area_l109_10929

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l109_10929


namespace NUMINAMATH_GPT_total_distance_traveled_l109_10992

theorem total_distance_traveled:
  let speed1 := 30
  let time1 := 4
  let speed2 := 35
  let time2 := 5
  let speed3 := 25
  let time3 := 6
  let total_time := 20
  let time1_3 := time1 + time2 + time3
  let time4 := total_time - time1_3
  let speed4 := 40

  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4

  let total_distance := distance1 + distance2 + distance3 + distance4

  total_distance = 645 :=
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l109_10992


namespace NUMINAMATH_GPT_determine_a_l109_10917

theorem determine_a (a x y : ℝ) (h : (a + 1) * x^(|a|) + y = -8) (h_linear : ∀ x y, (a + 1) * x^(|a|) + y = -8 → x ^ 1 = x): a = 1 :=
by 
  sorry

end NUMINAMATH_GPT_determine_a_l109_10917


namespace NUMINAMATH_GPT_solve_for_s_l109_10996

-- Definition of the given problem conditions
def parallelogram_sides_60_angle_sqrt_area (s : ℝ) :=
  ∃ (area : ℝ), (area = 27 * Real.sqrt 3) ∧
  (3 * s * s * Real.sqrt 3 = area)

-- Proof statement to demonstrate the equivalence of the theoretical and computed value of s
theorem solve_for_s (s : ℝ) : parallelogram_sides_60_angle_sqrt_area s → s = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_s_l109_10996


namespace NUMINAMATH_GPT_solution_exists_l109_10998

noncomputable def verifySolution (x y z : ℝ) : Prop := 
  x^2 - y = (z - 1)^2 ∧
  y^2 - z = (x - 1)^2 ∧
  z^2 - x = (y- 1)^2 

theorem solution_exists (x y z : ℝ) (h : verifySolution x y z) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x, y, z) = (-2.93122, 2.21124, 0.71998) ∨ 
  (x, y, z) = (2.21124, 0.71998, -2.93122) ∨ 
  (x, y, z) = (0.71998, -2.93122, 2.21124) :=
sorry

end NUMINAMATH_GPT_solution_exists_l109_10998


namespace NUMINAMATH_GPT_net_change_in_price_l109_10927

theorem net_change_in_price (P : ℝ) : 
  ((P * 0.75) * 1.2 = P * 0.9) → 
  ((P * 0.9 - P) / P = -0.1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_net_change_in_price_l109_10927


namespace NUMINAMATH_GPT_solve_for_x_l109_10980

theorem solve_for_x (x : ℕ) (hx : 1000^4 = 10^x) : x = 12 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l109_10980


namespace NUMINAMATH_GPT_asha_wins_probability_l109_10939

variable (p_lose p_tie p_win : ℚ)

theorem asha_wins_probability 
  (h_lose : p_lose = 3 / 7) 
  (h_tie : p_tie = 1 / 7) 
  (h_total : p_win + p_lose + p_tie = 1) : 
  p_win = 3 / 7 := by
  sorry

end NUMINAMATH_GPT_asha_wins_probability_l109_10939


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l109_10919

theorem necessary_but_not_sufficient_condition (x : ℝ) : (|x - 1| < 1 → x^2 - 5 * x < 0) ∧ (¬(x^2 - 5 * x < 0 → |x - 1| < 1)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l109_10919


namespace NUMINAMATH_GPT_fraction_equiv_l109_10906

theorem fraction_equiv (m n : ℚ) (h : m / n = 3 / 4) : (m + n) / n = 7 / 4 :=
sorry

end NUMINAMATH_GPT_fraction_equiv_l109_10906


namespace NUMINAMATH_GPT_find_a_in_triangle_l109_10950

theorem find_a_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : c = 3)
  (h2 : C = Real.pi / 3)
  (h3 : Real.sin B = 2 * Real.sin A)
  (h4 : a = 3) :
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_find_a_in_triangle_l109_10950


namespace NUMINAMATH_GPT_sales_overlap_l109_10934

-- Define the conditions
def bookstore_sale_days : List ℕ := [2, 6, 10, 14, 18, 22, 26, 30]
def shoe_store_sale_days : List ℕ := [1, 8, 15, 22, 29]

-- Define the statement to prove
theorem sales_overlap : (bookstore_sale_days ∩ shoe_store_sale_days).length = 1 := 
by
  sorry

end NUMINAMATH_GPT_sales_overlap_l109_10934


namespace NUMINAMATH_GPT_value_of_f_g6_minus_g_f6_l109_10930

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g6_minus_g_f6 : f (g 6) - g (f 6) = 48 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_g6_minus_g_f6_l109_10930


namespace NUMINAMATH_GPT_angle_acb_after_rotations_is_30_l109_10941

noncomputable def initial_angle : ℝ := 60
noncomputable def rotation_clockwise_540 : ℝ := -540
noncomputable def rotation_counterclockwise_90 : ℝ := 90
noncomputable def final_angle : ℝ := 30

theorem angle_acb_after_rotations_is_30 
  (initial_angle : ℝ)
  (rotation_clockwise_540 : ℝ)
  (rotation_counterclockwise_90 : ℝ) :
  final_angle = 30 :=
sorry

end NUMINAMATH_GPT_angle_acb_after_rotations_is_30_l109_10941


namespace NUMINAMATH_GPT_remaining_thumbtacks_in_each_can_l109_10928

-- Definitions based on the conditions:
def total_thumbtacks : ℕ := 450
def num_cans : ℕ := 3
def thumbtacks_per_board_tested : ℕ := 1
def total_boards_tested : ℕ := 120

-- Lean 4 Statement

theorem remaining_thumbtacks_in_each_can :
  ∀ (initial_thumbtacks_per_can remaining_thumbtacks_per_can : ℕ),
  initial_thumbtacks_per_can = (total_thumbtacks / num_cans) →
  remaining_thumbtacks_per_can = (initial_thumbtacks_per_can - (thumbtacks_per_board_tested * total_boards_tested)) →
  remaining_thumbtacks_per_can = 30 :=
by
  sorry

end NUMINAMATH_GPT_remaining_thumbtacks_in_each_can_l109_10928


namespace NUMINAMATH_GPT_Rachel_books_total_l109_10977

-- Define the conditions
def mystery_shelves := 6
def picture_shelves := 2
def scifi_shelves := 3
def bio_shelves := 4
def books_per_shelf := 9

-- Define the total number of books
def total_books := 
  mystery_shelves * books_per_shelf + 
  picture_shelves * books_per_shelf + 
  scifi_shelves * books_per_shelf + 
  bio_shelves * books_per_shelf

-- Statement of the problem
theorem Rachel_books_total : total_books = 135 := 
by
  -- Proof can be added here
  sorry

end NUMINAMATH_GPT_Rachel_books_total_l109_10977


namespace NUMINAMATH_GPT_problem_solution_l109_10946

theorem problem_solution :
  ∃ (b₂ b₃ b₄ b₅ b₆ b₇ : ℤ),
    (0 ≤ b₂ ∧ b₂ < 2) ∧
    (0 ≤ b₃ ∧ b₃ < 3) ∧
    (0 ≤ b₄ ∧ b₄ < 4) ∧
    (0 ≤ b₅ ∧ b₅ < 5) ∧
    (0 ≤ b₆ ∧ b₆ < 6) ∧
    (0 ≤ b₇ ∧ b₇ < 8) ∧
    (6 / 7 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040) ∧
    (b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11) :=
sorry

end NUMINAMATH_GPT_problem_solution_l109_10946


namespace NUMINAMATH_GPT_total_pets_remaining_l109_10976

def initial_counts := (7, 6, 4, 5, 3)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def morning_sales := (1, 2, 1, 0, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def afternoon_sales := (1, 1, 2, 3, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def returns := (0, 1, 0, 1, 1)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)

def calculate_remaining (initial_counts morning_sales afternoon_sales returns : Nat × Nat × Nat × Nat × Nat) : Nat :=
  let (p0, k0, r0, g0, c0) := initial_counts
  let (p1, k1, r1, g1, c1) := morning_sales
  let (p2, k2, r2, g2, c2) := afternoon_sales
  let (p3, k3, r3, g3, c3) := returns
  let remaining_puppies := p0 - p1 - p2 + p3
  let remaining_kittens := k0 - k1 - k2 + k3
  let remaining_rabbits := r0 - r1 - r2 + r3
  let remaining_guinea_pigs := g0 - g1 - g2 + g3
  let remaining_chameleons := c0 - c1 - c2 + c3
  remaining_puppies + remaining_kittens + remaining_rabbits + remaining_guinea_pigs + remaining_chameleons

theorem total_pets_remaining : calculate_remaining initial_counts morning_sales afternoon_sales returns = 15 := 
by
  simp [initial_counts, morning_sales, afternoon_sales, returns, calculate_remaining]
  sorry

end NUMINAMATH_GPT_total_pets_remaining_l109_10976


namespace NUMINAMATH_GPT_binary_to_decimal_110011_l109_10990

theorem binary_to_decimal_110011 : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_110011_l109_10990


namespace NUMINAMATH_GPT_initial_ratio_of_milk_water_l109_10923

theorem initial_ratio_of_milk_water (M W : ℝ) (H1 : M + W = 85) (H2 : M / (W + 5) = 3) : M / W = 27 / 7 :=
by sorry

end NUMINAMATH_GPT_initial_ratio_of_milk_water_l109_10923


namespace NUMINAMATH_GPT_volume_of_increased_box_l109_10993

theorem volume_of_increased_box {l w h : ℝ} (vol : l * w * h = 4860) (sa : l * w + w * h + l * h = 930) (sum_dim : l + w + h = 56) :
  (l + 2) * (w + 3) * (h + 1) = 5964 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_increased_box_l109_10993


namespace NUMINAMATH_GPT_inequality_8xyz_l109_10951

theorem inequality_8xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) > 8 * x * y * z := 
  by sorry

end NUMINAMATH_GPT_inequality_8xyz_l109_10951


namespace NUMINAMATH_GPT_rahim_average_price_l109_10909

def books_shop1 : ℕ := 50
def cost_shop1 : ℕ := 1000
def books_shop2 : ℕ := 40
def cost_shop2 : ℕ := 800

def total_books : ℕ := books_shop1 + books_shop2
def total_cost : ℕ := cost_shop1 + cost_shop2
def average_price_per_book : ℕ := total_cost / total_books

theorem rahim_average_price :
  average_price_per_book = 20 := by
  sorry

end NUMINAMATH_GPT_rahim_average_price_l109_10909


namespace NUMINAMATH_GPT_michael_hours_worked_l109_10965

def michael_hourly_rate := 7
def michael_overtime_rate := 2 * michael_hourly_rate
def work_hours := 40
def total_earnings := 320

theorem michael_hours_worked :
  (total_earnings = michael_hourly_rate * work_hours + michael_overtime_rate * (42 - work_hours)) :=
sorry

end NUMINAMATH_GPT_michael_hours_worked_l109_10965


namespace NUMINAMATH_GPT_solution_l109_10932

theorem solution (A B C : ℚ) (h1 : A + B = 10) (h2 : 2 * A = 3 * B + 5) (h3 : A * B * C = 120) :
  A = 7 ∧ B = 3 ∧ C = 40 / 7 := by
  sorry

end NUMINAMATH_GPT_solution_l109_10932


namespace NUMINAMATH_GPT_parabola_centroid_locus_l109_10949

/-- Let P_0 be a parabola defined by the equation y = m * x^2. 
    Let A and B be points on P_0 such that the tangents at A and B are perpendicular. 
    Let G be the centroid of the triangle formed by A, B, and the vertex of P_0.
    Let P_n be the nth derived parabola.
    Prove that the equation of P_n is y = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n). -/
theorem parabola_centroid_locus (n : ℕ) (m : ℝ) 
  (h_pos_m : 0 < m) :
  ∃ P_n : ℝ → ℝ, 
    ∀ x : ℝ, P_n x = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n) :=
sorry

end NUMINAMATH_GPT_parabola_centroid_locus_l109_10949
