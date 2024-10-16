import Mathlib

namespace NUMINAMATH_CALUDE_percentage_calculation_l2077_207795

theorem percentage_calculation (y : ℝ) : 
  0.11 * y = 0.3 * (0.7 * y) - 0.1 * y := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2077_207795


namespace NUMINAMATH_CALUDE_smallest_other_integer_l2077_207754

theorem smallest_other_integer (m n x : ℕ) : 
  m = 30 →
  x > 0 →
  Nat.gcd m n = x + 1 →
  Nat.lcm m n = x * (x + 1) →
  ∃ (n_min : ℕ), n_min = 6 ∧ ∀ (n' : ℕ), (
    Nat.gcd m n' = x + 1 ∧
    Nat.lcm m n' = x * (x + 1) →
    n' ≥ n_min
  ) := by sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l2077_207754


namespace NUMINAMATH_CALUDE_particle_max_elevation_l2077_207703

noncomputable def s (t : ℝ) : ℝ := 200 * t - 17 * t^2 - 3 * t^3

theorem particle_max_elevation :
  ∃ (max_height : ℝ), 
    (∀ t : ℝ, t ≥ 0 → s t ≤ max_height) ∧ 
    (∃ t : ℝ, t ≥ 0 ∧ s t = max_height) ∧
    (abs (max_height - 368.1) < 0.1) := by
  sorry

end NUMINAMATH_CALUDE_particle_max_elevation_l2077_207703


namespace NUMINAMATH_CALUDE_points_above_line_t_range_l2077_207750

def P (t : ℝ) : ℝ × ℝ := (1, t)
def Q (t : ℝ) : ℝ × ℝ := (t^2, t - 1)

def above_line (p : ℝ × ℝ) : Prop := p.1 + p.2 - 1 > 0

theorem points_above_line_t_range :
  ∀ t : ℝ, (above_line (P t) ∧ above_line (Q t)) ↔ t > 1 := by
sorry

end NUMINAMATH_CALUDE_points_above_line_t_range_l2077_207750


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2077_207715

theorem interest_rate_calculation (P : ℝ) (t : ℝ) (diff : ℝ) (r : ℝ) : 
  P = 5100 → 
  t = 2 → 
  P * ((1 + r) ^ t - 1) - P * r * t = diff → 
  diff = 51 → 
  r = 0.1 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2077_207715


namespace NUMINAMATH_CALUDE_original_flock_size_l2077_207781

/-- Represents a flock of sheep --/
structure Flock where
  rams : ℕ
  ewes : ℕ

/-- The original flock of sheep --/
def original_flock : Flock := sorry

/-- The flock after one ram runs away --/
def flock_minus_ram : Flock := 
  { rams := original_flock.rams - 1, ewes := original_flock.ewes }

/-- The flock after the ram returns and one ewe runs away --/
def flock_minus_ewe : Flock := 
  { rams := original_flock.rams, ewes := original_flock.ewes - 1 }

/-- The theorem to be proved --/
theorem original_flock_size : 
  (flock_minus_ram.rams : ℚ) / flock_minus_ram.ewes = 7 / 5 ∧
  (flock_minus_ewe.rams : ℚ) / flock_minus_ewe.ewes = 5 / 3 →
  original_flock.rams + original_flock.ewes = 25 := by
  sorry


end NUMINAMATH_CALUDE_original_flock_size_l2077_207781


namespace NUMINAMATH_CALUDE_m_range_l2077_207727

def f (m : ℝ) (x : ℝ) := 2*x^2 - 2*(m-2)*x + 3*m - 1

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop :=
  m + 1 > 0 ∧ 9 - m > m + 1

def prop_p (m : ℝ) : Prop := is_increasing (f m) 1 2

def prop_q (m : ℝ) : Prop := is_ellipse_with_foci_on_y_axis m

theorem m_range (m : ℝ) 
  (h1 : prop_p m ∨ prop_q m) 
  (h2 : ¬(prop_p m ∧ prop_q m)) 
  (h3 : ¬¬(prop_p m)) : 
  m ≤ -1 ∨ m = 4 := by sorry

end NUMINAMATH_CALUDE_m_range_l2077_207727


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertex_probability_l2077_207762

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Nat := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def AdjacentVertices : Nat := 2

/-- The probability of choosing two distinct adjacent vertices in a decagon -/
theorem decagon_adjacent_vertex_probability : 
  (AdjacentVertices : ℚ) / (Decagon - 1 : ℚ) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertex_probability_l2077_207762


namespace NUMINAMATH_CALUDE_magnitude_v_l2077_207737

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * I) (h2 : Complex.abs u = Real.sqrt 34) : 
  Complex.abs v = (25 * Real.sqrt 34) / 34 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_v_l2077_207737


namespace NUMINAMATH_CALUDE_dave_car_count_l2077_207708

theorem dave_car_count (store1 store2 store3 store4 store5 : ℕ) 
  (h1 : store2 = 14)
  (h2 : store3 = 14)
  (h3 : store4 = 21)
  (h4 : store5 = 25)
  (h5 : (store1 + store2 + store3 + store4 + store5) / 5 = 208/10) :
  store1 = 30 := by
sorry

end NUMINAMATH_CALUDE_dave_car_count_l2077_207708


namespace NUMINAMATH_CALUDE_arithmetic_24_l2077_207758

def numbers : List ℕ := [8, 8, 8, 10]

inductive ArithExpr
  | Num : ℕ → ArithExpr
  | Add : ArithExpr → ArithExpr → ArithExpr
  | Sub : ArithExpr → ArithExpr → ArithExpr
  | Mul : ArithExpr → ArithExpr → ArithExpr
  | Div : ArithExpr → ArithExpr → ArithExpr

def eval : ArithExpr → ℕ
  | ArithExpr.Num n => n
  | ArithExpr.Add e1 e2 => eval e1 + eval e2
  | ArithExpr.Sub e1 e2 => eval e1 - eval e2
  | ArithExpr.Mul e1 e2 => eval e1 * eval e2
  | ArithExpr.Div e1 e2 => eval e1 / eval e2

def uses_all_numbers (expr : ArithExpr) (nums : List ℕ) : Prop := sorry

theorem arithmetic_24 : 
  ∃ (expr : ArithExpr), uses_all_numbers expr numbers ∧ eval expr = 24 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_24_l2077_207758


namespace NUMINAMATH_CALUDE_polynomial_is_perfect_square_l2077_207770

theorem polynomial_is_perfect_square (x : ℝ) : 
  ∃ (t u : ℝ), (49/4 : ℝ) * x^2 + 21 * x + 9 = (t * x + u)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_is_perfect_square_l2077_207770


namespace NUMINAMATH_CALUDE_money_left_l2077_207735

def initial_amount : ℕ := 48
def num_books : ℕ := 5
def book_cost : ℕ := 2

theorem money_left : initial_amount - (num_books * book_cost) = 38 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l2077_207735


namespace NUMINAMATH_CALUDE_task_completion_probability_l2077_207728

theorem task_completion_probability (p_task1 p_task1_not_task2 : ℝ) 
  (h1 : p_task1 = 3/8)
  (h2 : p_task1_not_task2 = 0.15)
  (h3 : 0 ≤ p_task1 ∧ p_task1 ≤ 1)
  (h4 : 0 ≤ p_task1_not_task2 ∧ p_task1_not_task2 ≤ 1) :
  ∃ p_task2 : ℝ, p_task2 = 0.6 ∧ 0 ≤ p_task2 ∧ p_task2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_probability_l2077_207728


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2077_207730

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2077_207730


namespace NUMINAMATH_CALUDE_f_difference_at_five_l2077_207798

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x

-- State the theorem
theorem f_difference_at_five : f 5 - f (-5) = 50 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_five_l2077_207798


namespace NUMINAMATH_CALUDE_factorization_equality_l2077_207705

theorem factorization_equality (a b : ℝ) : 4 * a^2 * b - b = b * (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2077_207705


namespace NUMINAMATH_CALUDE_max_z_value_l2077_207760

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x*y + y*z + z*x = 3) :
  z ≤ 13/3 :=
by sorry

end NUMINAMATH_CALUDE_max_z_value_l2077_207760


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l2077_207799

theorem least_four_digit_multiple : ∀ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) → 
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0) → 
  1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l2077_207799


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2077_207734

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 500)
  (h3 : crossing_time = 8) :
  (train_length + bridge_length) / crossing_time = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2077_207734


namespace NUMINAMATH_CALUDE_range_of_a_l2077_207789

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x + 2| ≤ 3) → -5 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2077_207789


namespace NUMINAMATH_CALUDE_power_of_power_five_l2077_207755

theorem power_of_power_five : (5^2)^4 = 390625 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_five_l2077_207755


namespace NUMINAMATH_CALUDE_charity_event_arrangements_l2077_207707

/-- The number of ways to arrange volunteers for a 3-day charity event -/
def charity_arrangements (total_volunteers : ℕ) (day1_needed : ℕ) (day2_needed : ℕ) (day3_needed : ℕ) : ℕ :=
  Nat.choose total_volunteers day1_needed *
  Nat.choose (total_volunteers - day1_needed) day2_needed *
  Nat.choose (total_volunteers - day1_needed - day2_needed) day3_needed

/-- Theorem stating that the number of arrangements for the given conditions is 60 -/
theorem charity_event_arrangements :
  charity_arrangements 6 1 2 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_charity_event_arrangements_l2077_207707


namespace NUMINAMATH_CALUDE_linda_spent_correct_l2077_207784

/-- The total amount Linda spent on school supplies -/
def linda_total_spent : ℝ := 6.80

/-- The cost of a single notebook -/
def notebook_cost : ℝ := 1.20

/-- The number of notebooks Linda bought -/
def notebook_quantity : ℕ := 3

/-- The cost of a box of pencils -/
def pencil_box_cost : ℝ := 1.50

/-- The cost of a box of pens -/
def pen_box_cost : ℝ := 1.70

/-- Theorem stating that the total amount Linda spent is correct -/
theorem linda_spent_correct :
  linda_total_spent = notebook_cost * (notebook_quantity : ℝ) + pencil_box_cost + pen_box_cost := by
  sorry

end NUMINAMATH_CALUDE_linda_spent_correct_l2077_207784


namespace NUMINAMATH_CALUDE_remainder_2021_div_102_l2077_207709

theorem remainder_2021_div_102 : 2021 % 102 = 83 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2021_div_102_l2077_207709


namespace NUMINAMATH_CALUDE_decryption_result_l2077_207772

/-- Represents an encrypted text -/
def EncryptedText := String

/-- Represents a decrypted text -/
def DecryptedText := String

/-- The encryption method used for the original message -/
def encryptionMethod (original : String) (encrypted : EncryptedText) : Prop :=
  encrypted.toList.filter (· ∈ original.toList) = original.toList

/-- The decryption function -/
noncomputable def decrypt (text : EncryptedText) : DecryptedText :=
  sorry

/-- Theorem stating the decryption results -/
theorem decryption_result 
  (text1 text2 text3 : EncryptedText)
  (h1 : encryptionMethod "МОСКВА" "ЙМЫВОТСБЛКЪГВЦАЯЯ")
  (h2 : encryptionMethod "МОСКВА" "УКМАПОЧСРКЩВЗАХ")
  (h3 : encryptionMethod "МОСКВА" "ШМФЭОГЧСЙЪКФЬВЫЕАКК")
  (h4 : text1 = "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ")
  (h5 : text2 = "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП")
  (h6 : text3 = "РТПАИОМВСВТИЕОБПРОЕННИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК") :
  (decrypt text1 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧
   decrypt text2 = "С ЧИСТОЙ СОВЕСТЬЮ" ∧
   decrypt text3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ") :=
by sorry

end NUMINAMATH_CALUDE_decryption_result_l2077_207772


namespace NUMINAMATH_CALUDE_last_three_digits_of_9_pow_107_l2077_207720

theorem last_three_digits_of_9_pow_107 : 9^107 % 1000 = 969 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_9_pow_107_l2077_207720


namespace NUMINAMATH_CALUDE_factorization_sum_l2077_207739

theorem factorization_sum (a b : ℤ) : 
  (∀ x : ℝ, 16 * x^2 - 106 * x - 105 = (8 * x + a) * (2 * x + b)) → 
  a + 2 * b = -23 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l2077_207739


namespace NUMINAMATH_CALUDE_y_minimized_at_b_over_2_l2077_207747

variable (a b : ℝ)

def y (x : ℝ) := (x - a)^2 + (x - b)^2 + 2*(a - b)*x

theorem y_minimized_at_b_over_2 :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y a b x_min ≤ y a b x ∧ x_min = b/2 :=
sorry

end NUMINAMATH_CALUDE_y_minimized_at_b_over_2_l2077_207747


namespace NUMINAMATH_CALUDE_eight_point_five_million_scientific_notation_l2077_207718

theorem eight_point_five_million_scientific_notation :
  (8.5 * 1000000 : ℝ) = 8.5 * (10 ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_eight_point_five_million_scientific_notation_l2077_207718


namespace NUMINAMATH_CALUDE_head_start_value_l2077_207706

-- Define the race parameters
def race_length : ℝ := 142

-- Define the speeds of A and B
def speed_ratio : ℝ := 2

-- Define the head start function
def head_start (s : ℝ) : Prop :=
  ∀ (v : ℝ), v > 0 → (race_length / (speed_ratio * v)) = ((race_length - s) / v)

-- Theorem statement
theorem head_start_value : head_start 71 := by
  sorry

end NUMINAMATH_CALUDE_head_start_value_l2077_207706


namespace NUMINAMATH_CALUDE_bruce_grapes_purchase_l2077_207743

theorem bruce_grapes_purchase (grape_price : ℝ) (mango_price : ℝ) (mango_quantity : ℝ) (total_paid : ℝ) :
  grape_price = 70 →
  mango_price = 55 →
  mango_quantity = 10 →
  total_paid = 1110 →
  (total_paid - mango_price * mango_quantity) / grape_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_bruce_grapes_purchase_l2077_207743


namespace NUMINAMATH_CALUDE_intersection_properties_l2077_207719

/-- Given a line y = a intersecting two curves, prove properties of intersection points -/
theorem intersection_properties (a : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (∀ x, x/Real.exp x = a ↔ x = x₁ ∨ x = x₂) →  -- y = x/e^x intersects y = a at x₁ and x₂
  (∀ x, Real.log x/x = a ↔ x = x₂ ∨ x = x₃) →  -- y = ln(x)/x intersects y = a at x₂ and x₃
  0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ →                 -- order of x₁, x₂, x₃
  (x₂ = a * Real.exp x₂ ∧                      -- Statement A
   x₃ = Real.exp x₂ ∧                          -- Statement C
   x₁ + x₃ > 2 * x₂)                           -- Statement D
:= by sorry

end NUMINAMATH_CALUDE_intersection_properties_l2077_207719


namespace NUMINAMATH_CALUDE_play_attendance_l2077_207717

theorem play_attendance (total_people : ℕ) (adult_price child_price : ℕ) (total_receipts : ℕ) :
  total_people = 610 →
  adult_price = 2 →
  child_price = 1 →
  total_receipts = 960 →
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_receipts ∧
    adults = 350 :=
by sorry

end NUMINAMATH_CALUDE_play_attendance_l2077_207717


namespace NUMINAMATH_CALUDE_john_driving_distance_l2077_207778

/-- Represents the efficiency of John's car in miles per gallon -/
def car_efficiency : ℝ := 40

/-- Represents the current price of gas in dollars per gallon -/
def gas_price : ℝ := 5

/-- Represents the amount of money John has to spend on gas in dollars -/
def available_money : ℝ := 25

/-- Theorem stating that John can drive exactly 200 miles with the given conditions -/
theorem john_driving_distance : 
  (available_money / gas_price) * car_efficiency = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_driving_distance_l2077_207778


namespace NUMINAMATH_CALUDE_tire_usage_calculation_tire_usage_proof_l2077_207753

/-- Calculates the miles each tire was used given the total distance and tire usage pattern. -/
theorem tire_usage_calculation (total_distance : ℕ) (first_part_distance : ℕ) (second_part_distance : ℕ) 
  (total_tires : ℕ) (tires_used_first_part : ℕ) (tires_used_second_part : ℕ) : ℕ :=
  let total_tire_miles := first_part_distance * tires_used_first_part + second_part_distance * tires_used_second_part
  total_tire_miles / total_tires

/-- Proves that each tire was used for 38,571 miles given the specific conditions of the problem. -/
theorem tire_usage_proof : 
  tire_usage_calculation 50000 40000 10000 7 5 7 = 38571 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_calculation_tire_usage_proof_l2077_207753


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l2077_207736

theorem quadratic_form_minimum : ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -4.45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l2077_207736


namespace NUMINAMATH_CALUDE_count_is_58_l2077_207751

/-- A function that generates all permutations of a list -/
def permutations (l : List ℕ) : List (List ℕ) :=
  sorry

/-- A function that converts a list of digits to a number -/
def list_to_number (l : List ℕ) : ℕ :=
  sorry

/-- The set of digits we're working with -/
def digits : List ℕ := [1, 2, 3, 4, 5]

/-- All possible five-digit numbers from the given digits -/
def all_numbers : List ℕ :=
  (permutations digits).map list_to_number

/-- The count of numbers satisfying our conditions -/
def count_numbers : ℕ :=
  (all_numbers.filter (λ n => n > 23145 ∧ n < 43521)).length

theorem count_is_58 : count_numbers = 58 :=
  sorry

end NUMINAMATH_CALUDE_count_is_58_l2077_207751


namespace NUMINAMATH_CALUDE_trajectory_theorem_l2077_207733

def trajectory_problem (R h : ℝ) (θ : ℝ) : Prop :=
  let r₁ := R * Real.cos θ
  let r₂ := (R + h) * Real.cos θ
  let s := 2 * Real.pi * r₂ - 2 * Real.pi * r₁
  s = h ∧ θ = Real.arccos (1 / (2 * Real.pi))

theorem trajectory_theorem :
  ∀ (R h : ℝ), R > 0 → h > 0 → ∃ θ : ℝ, trajectory_problem R h θ :=
sorry

end NUMINAMATH_CALUDE_trajectory_theorem_l2077_207733


namespace NUMINAMATH_CALUDE_programmers_typing_speed_l2077_207702

/-- The number of programmers --/
def num_programmers : ℕ := 10

/-- The number of lines typed in 60 minutes --/
def lines_in_60_min : ℕ := 60

/-- The duration in minutes for which we want to calculate the lines typed --/
def target_duration : ℕ := 10

/-- Theorem stating that the programmers can type 100 lines in 10 minutes --/
theorem programmers_typing_speed :
  (num_programmers * lines_in_60_min * target_duration) / 60 = 100 := by
  sorry

end NUMINAMATH_CALUDE_programmers_typing_speed_l2077_207702


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_9_4_l2077_207711

theorem smallest_five_digit_mod_9_4 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n ≡ 4 [ZMOD 9] → 
    10003 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_9_4_l2077_207711


namespace NUMINAMATH_CALUDE_sum_of_baby_ages_theorem_l2077_207740

/- Define the ages of the animals -/
def lioness_age : ℕ := 12
def hyena_age : ℕ := lioness_age / 2
def leopard_age : ℕ := 3 * hyena_age

/- Define the ages of the babies -/
def lioness_baby_age : ℕ := lioness_age / 2
def hyena_baby_age : ℕ := hyena_age / 2
def leopard_baby_age : ℕ := leopard_age / 2

/- Define the sum of the babies' ages after 5 years -/
def sum_of_baby_ages_after_5_years : ℕ := 
  (lioness_baby_age + 5) + (hyena_baby_age + 5) + (leopard_baby_age + 5)

theorem sum_of_baby_ages_theorem : sum_of_baby_ages_after_5_years = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_baby_ages_theorem_l2077_207740


namespace NUMINAMATH_CALUDE_max_t_geq_pi_l2077_207759

noncomputable def f (x : ℝ) : ℝ := Real.sin x / x

theorem max_t_geq_pi (t : ℝ) (h : ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < t → f x₁ > f x₂) :
  t ≥ π :=
sorry

end NUMINAMATH_CALUDE_max_t_geq_pi_l2077_207759


namespace NUMINAMATH_CALUDE_initial_points_count_l2077_207700

/-- The number of points after one operation -/
def points_after_one_op (n : ℕ) : ℕ := 2 * n - 1

/-- The number of points after two operations -/
def points_after_two_ops (n : ℕ) : ℕ := 2 * (points_after_one_op n) - 1

/-- The number of points after three operations -/
def points_after_three_ops (n : ℕ) : ℕ := 2 * (points_after_two_ops n) - 1

/-- 
Theorem: If we start with n points on a line, perform the operation of adding a point 
between each pair of neighboring points three times, and end up with 65 points, 
then n must be equal to 9.
-/
theorem initial_points_count : points_after_three_ops 9 = 65 ∧ 
  (∀ m : ℕ, points_after_three_ops m = 65 → m = 9) := by
  sorry

end NUMINAMATH_CALUDE_initial_points_count_l2077_207700


namespace NUMINAMATH_CALUDE_find_number_l2077_207782

theorem find_number : ∃ X : ℝ, (50 : ℝ) = 0.2 * X + 47 ∧ X = 15 := by sorry

end NUMINAMATH_CALUDE_find_number_l2077_207782


namespace NUMINAMATH_CALUDE_power_three_mod_ten_l2077_207746

theorem power_three_mod_ten : 3^24 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_ten_l2077_207746


namespace NUMINAMATH_CALUDE_min_value_of_f_l2077_207783

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + 2

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), m = 50/27 ∧ ∀ (x : ℝ), x > 0 → f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2077_207783


namespace NUMINAMATH_CALUDE_civil_servant_dispatch_l2077_207712

theorem civil_servant_dispatch (m n k : ℕ) (hm : m = 5) (hn : n = 4) (hk : k = 3) :
  (k.factorial * (Nat.choose (m + n) k - Nat.choose m k - Nat.choose n k)) = 420 :=
by sorry

end NUMINAMATH_CALUDE_civil_servant_dispatch_l2077_207712


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2077_207773

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 200 →
  first_discount = 20 →
  final_price = 152 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 5 :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l2077_207773


namespace NUMINAMATH_CALUDE_total_time_to_grandmaster_l2077_207722

/-- Time spent on learning basic chess rules (in hours) -/
def basic_rules : ℝ := 2

/-- Factor for intermediate level time compared to basic rules -/
def intermediate_factor : ℝ := 75

/-- Factor for expert level time compared to combined basic and intermediate -/
def expert_factor : ℝ := 50

/-- Factor for master level time compared to expert level -/
def master_factor : ℝ := 30

/-- Percentage of intermediate level time spent on endgame exercises -/
def endgame_percentage : ℝ := 0.25

/-- Factor for middle game study compared to endgame exercises -/
def middle_game_factor : ℝ := 2

/-- Percentage of expert level time spent on mentoring -/
def mentoring_percentage : ℝ := 0.5

/-- Theorem: The total time James spent to become a chess grandmaster -/
theorem total_time_to_grandmaster :
  let intermediate := basic_rules * intermediate_factor
  let expert := expert_factor * (basic_rules + intermediate)
  let master := master_factor * expert
  let endgame := endgame_percentage * intermediate
  let middle_game := middle_game_factor * endgame
  let mentoring := mentoring_percentage * expert
  basic_rules + intermediate + expert + master + endgame + middle_game + mentoring = 235664.5 := by
sorry

end NUMINAMATH_CALUDE_total_time_to_grandmaster_l2077_207722


namespace NUMINAMATH_CALUDE_gift_box_wrapping_l2077_207701

theorem gift_box_wrapping (total_ribbon : ℝ) (ribbon_per_box : ℝ) :
  total_ribbon = 25 →
  ribbon_per_box = 1.6 →
  ⌊total_ribbon / ribbon_per_box⌋ = 15 := by
  sorry

end NUMINAMATH_CALUDE_gift_box_wrapping_l2077_207701


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2077_207793

theorem contrapositive_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, ¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2077_207793


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l2077_207749

theorem mistaken_multiplication (x : ℚ) : 
  6 * x = 12 → 7 * x = 14 := by
sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l2077_207749


namespace NUMINAMATH_CALUDE_equilateral_triangle_vertex_l2077_207765

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (a b c : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = (b.x - c.x)^2 + (b.y - c.y)^2 ∧
  (b.x - c.x)^2 + (b.y - c.y)^2 = (c.x - a.x)^2 + (c.y - a.y)^2

/-- Checks if a point is on the altitude from another point to a line segment -/
def isOnAltitude (a d : Point) (b c : Point) : Prop :=
  (d.x - b.x) * (c.x - b.x) + (d.y - b.y) * (c.y - b.y) = 0 ∧
  (a.x - d.x) * (c.x - b.x) + (a.y - d.y) * (c.y - b.y) = 0

theorem equilateral_triangle_vertex (a b d : Point) : 
  a = Point.mk 10 4 →
  b = Point.mk 1 (-5) →
  d = Point.mk 0 (-2) →
  ∃ c : Point, 
    isEquilateral a b c ∧ 
    isOnAltitude a d b c ∧ 
    c = Point.mk (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_vertex_l2077_207765


namespace NUMINAMATH_CALUDE_product_decomposition_l2077_207748

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def options : List ℕ := [2986, 2858, 2672, 2754]

theorem product_decomposition :
  ∃! (product : ℕ) (a b : ℕ), 
    product ∈ options ∧
    is_two_digit a ∧
    is_three_digit b ∧
    product = a * b :=
sorry

end NUMINAMATH_CALUDE_product_decomposition_l2077_207748


namespace NUMINAMATH_CALUDE_max_profit_achieved_l2077_207723

/-- Represents the selling and cost information for peanuts and tea --/
structure ProductInfo where
  peanut_price : ℝ
  tea_price : ℝ
  peanut_cost : ℝ
  tea_cost : ℝ

/-- Represents the sales constraints --/
structure SalesConstraints where
  total_quantity : ℝ
  max_cost : ℝ

/-- Calculates the profit for given quantities of peanuts and tea --/
def calculate_profit (info : ProductInfo) (peanut_qty : ℝ) (tea_qty : ℝ) : ℝ :=
  (info.peanut_price - info.peanut_cost) * peanut_qty +
  (info.tea_price - info.tea_cost) * tea_qty

/-- Checks if the sales quantities satisfy the given constraints --/
def satisfies_constraints (constraints : SalesConstraints) (info : ProductInfo)
    (peanut_qty : ℝ) (tea_qty : ℝ) : Prop :=
  peanut_qty + tea_qty = constraints.total_quantity ∧
  info.peanut_cost * peanut_qty + info.tea_cost * tea_qty ≤ constraints.max_cost ∧
  peanut_qty ≤ 2 * tea_qty

/-- Main theorem stating that the maximum profit is achieved at specific quantities --/
theorem max_profit_achieved (info : ProductInfo) (constraints : SalesConstraints) :
    info.tea_price = info.peanut_price + 40 →
    50 * info.peanut_price = 10 * info.tea_price →
    info.peanut_cost = 6 →
    info.tea_cost = 36 →
    constraints.total_quantity = 60 →
    constraints.max_cost = 1260 →
    ∃ (max_profit : ℝ),
      max_profit = 540 ∧
      ∀ (peanut_qty tea_qty : ℝ),
        satisfies_constraints constraints info peanut_qty tea_qty →
        calculate_profit info peanut_qty tea_qty ≤ max_profit ∧
        (calculate_profit info peanut_qty tea_qty = max_profit ↔
         peanut_qty = 30 ∧ tea_qty = 30) := by
  sorry

end NUMINAMATH_CALUDE_max_profit_achieved_l2077_207723


namespace NUMINAMATH_CALUDE_hyperbola_range_theorem_l2077_207742

/-- The range of m for which the equation represents a hyperbola -/
def hyperbola_range : Set ℝ := Set.union (Set.Ioo (-1) 1) (Set.Ioi 2)

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1 ∧
  ((|m| - 1 > 0 ∧ 2 - m < 0) ∨ (|m| - 1 < 0 ∧ 2 - m > 0))

/-- Theorem stating the range of m for which the equation represents a hyperbola -/
theorem hyperbola_range_theorem :
  ∀ m : ℝ, is_hyperbola m ↔ m ∈ hyperbola_range :=
sorry

end NUMINAMATH_CALUDE_hyperbola_range_theorem_l2077_207742


namespace NUMINAMATH_CALUDE_loan_duration_proof_l2077_207710

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

theorem loan_duration_proof (principal rate total_returned : ℝ) 
  (h1 : principal = 5396.103896103896)
  (h2 : rate = 0.06)
  (h3 : total_returned = 8310) :
  ∃ t : ℝ, t = 9 ∧ total_returned = principal + simple_interest principal rate t := by
  sorry

#eval simple_interest 5396.103896103896 0.06 9

end NUMINAMATH_CALUDE_loan_duration_proof_l2077_207710


namespace NUMINAMATH_CALUDE_possible_values_of_d_l2077_207780

theorem possible_values_of_d (a b c d : ℕ) 
  (h : (a * d - 1) / (a + 1) + (b * d - 1) / (b + 1) + (c * d - 1) / (c + 1) = d) :
  d = 1 ∨ d = 2 ∨ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_d_l2077_207780


namespace NUMINAMATH_CALUDE_common_chord_equation_l2077_207774

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the common chord line
def common_chord (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem common_chord_equation :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2077_207774


namespace NUMINAMATH_CALUDE_function_identity_l2077_207779

theorem function_identity (f : ℕ → ℕ) : 
  (∀ m n : ℕ, f (m + f n) = f (f m) + f n) → 
  (∀ n : ℕ, f n = n) := by
sorry

end NUMINAMATH_CALUDE_function_identity_l2077_207779


namespace NUMINAMATH_CALUDE_bus_profit_analysis_l2077_207731

/-- Represents the daily profit of a bus company -/
def daily_profit (x : ℕ) : ℤ :=
  2 * x - 600

theorem bus_profit_analysis :
  (∀ x : ℕ, x ≥ 300 → daily_profit x ≥ 0) ∧
  (∀ x : ℕ, daily_profit x = 2 * x - 600) ∧
  (daily_profit 800 = 1000) :=
sorry

end NUMINAMATH_CALUDE_bus_profit_analysis_l2077_207731


namespace NUMINAMATH_CALUDE_pyramid_blocks_l2077_207785

/-- Calculates the number of blocks in a pyramid layer given the number in the layer above -/
def blocks_in_layer (blocks_above : ℕ) : ℕ := 3 * blocks_above

/-- Calculates the total number of blocks in a pyramid with the given number of layers -/
def total_blocks (layers : ℕ) : ℕ :=
  match layers with
  | 0 => 0
  | n + 1 => (blocks_in_layer^[n] 1) + total_blocks n

theorem pyramid_blocks :
  total_blocks 4 = 40 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_blocks_l2077_207785


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_implication_is_true_l2077_207777

theorem negation_of_absolute_value_implication_is_true : 
  (∃ x : ℝ, (|x| ≤ 1 ∧ x > 1) ∨ (|x| > 1 ∧ x ≤ 1)) = False :=
by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_implication_is_true_l2077_207777


namespace NUMINAMATH_CALUDE_paris_travel_distance_l2077_207792

theorem paris_travel_distance (total_distance train_distance bus_distance cab_distance : ℝ) : 
  total_distance = 500 ∧
  bus_distance = train_distance / 2 ∧
  cab_distance = bus_distance / 3 ∧
  total_distance = train_distance + bus_distance + cab_distance →
  train_distance = 300 := by
sorry

end NUMINAMATH_CALUDE_paris_travel_distance_l2077_207792


namespace NUMINAMATH_CALUDE_problem_solution_l2077_207768

theorem problem_solution :
  -- Part 1
  (let a : ℤ := 2
   let b : ℤ := -1
   (3 * a^2 * b + (1/4) * a * b^2 - (3/4) * a * b^2 + a^2 * b) = -17) ∧
  -- Part 2
  (∀ x y : ℝ, ∃ a b : ℝ,
    (2*x^2 + a*x - y + 6) - (2*b*x^2 - 3*x + 5*y - 1) = 0 →
    5*a*b^2 - (a^2*b + 2*(a^2*b - 3*a*b^2)) = -60) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2077_207768


namespace NUMINAMATH_CALUDE_internal_diagonal_cubes_l2077_207771

def cuboid_diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

theorem internal_diagonal_cubes :
  cuboid_diagonal_cubes 168 350 390 = 880 := by
  sorry

end NUMINAMATH_CALUDE_internal_diagonal_cubes_l2077_207771


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2077_207763

/-- An isosceles triangle with two sides of length 8 cm and perimeter 25 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length 
  (a b c : ℝ) 
  (h_isosceles : a = b) 
  (h_congruent_sides : a = 8) 
  (h_perimeter : a + b + c = 25) : 
  c = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2077_207763


namespace NUMINAMATH_CALUDE_order_of_abc_l2077_207788

noncomputable def a : ℝ := 2 + Real.sqrt 3
noncomputable def b : ℝ := 1 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 + Real.sqrt 5

theorem order_of_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l2077_207788


namespace NUMINAMATH_CALUDE_green_balls_removal_l2077_207704

theorem green_balls_removal (total : ℕ) (red_percent : ℚ) (green_removed : ℕ) :
  total = 150 →
  red_percent = 2/5 →
  green_removed = 75 →
  (red_percent * ↑total : ℚ) / (↑total - ↑green_removed : ℚ) = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_green_balls_removal_l2077_207704


namespace NUMINAMATH_CALUDE_ball_probabilities_l2077_207714

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  total : ℕ
  red : ℕ
  black : ℕ
  white : ℕ
  green : ℕ

/-- The given ball counts in the problem -/
def problemCounts : BallCounts := {
  total := 12,
  red := 5,
  black := 4,
  white := 2,
  green := 1
}

/-- Calculates the probability of drawing a red or black ball -/
def probRedOrBlack (counts : BallCounts) : ℚ :=
  (counts.red + counts.black : ℚ) / counts.total

/-- Calculates the probability of drawing at least one red ball when two balls are drawn -/
def probAtLeastOneRed (counts : BallCounts) : ℚ :=
  let totalWays := counts.total * (counts.total - 1) / 2
  let oneRedWays := counts.red * (counts.total - counts.red)
  let twoRedWays := counts.red * (counts.red - 1) / 2
  (oneRedWays + twoRedWays : ℚ) / totalWays

theorem ball_probabilities (counts : BallCounts) 
    (h_total : counts.total = 12)
    (h_red : counts.red = 5)
    (h_black : counts.black = 4)
    (h_white : counts.white = 2)
    (h_green : counts.green = 1) :
    probRedOrBlack counts = 3/4 ∧ probAtLeastOneRed counts = 15/22 := by
  sorry

#eval probRedOrBlack problemCounts
#eval probAtLeastOneRed problemCounts

end NUMINAMATH_CALUDE_ball_probabilities_l2077_207714


namespace NUMINAMATH_CALUDE_product_from_lcm_hcf_l2077_207726

theorem product_from_lcm_hcf (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 600) 
  (h_hcf : Nat.gcd a b = 30) : 
  a * b = 18000 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_hcf_l2077_207726


namespace NUMINAMATH_CALUDE_cubic_sum_identity_l2077_207790

theorem cubic_sum_identity (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h_sum : a + b + c = d) :
  (a^3 + b^3 + c^3 - 3*a*b*c) / (a*b*c) = d * (a^2 + b^2 + c^2 - a*b - a*c - b*c) / (a*b*c) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_identity_l2077_207790


namespace NUMINAMATH_CALUDE_cos_four_pi_thirds_minus_alpha_l2077_207713

theorem cos_four_pi_thirds_minus_alpha (α : ℝ) 
  (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (4 * π / 3 - α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_pi_thirds_minus_alpha_l2077_207713


namespace NUMINAMATH_CALUDE_fraction_power_product_l2077_207786

theorem fraction_power_product :
  (8 / 9 : ℚ)^3 * (5 / 3 : ℚ)^3 = 64000 / 19683 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l2077_207786


namespace NUMINAMATH_CALUDE_trajectory_is_hyperbola_l2077_207752

-- Define the two fixed circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the moving circle
def movingCircle (cx cy r : ℝ) : Prop := ∀ (x y : ℝ), (x - cx)^2 + (y - cy)^2 = r^2

-- Define the tangency condition
def isTangent (cx cy r : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), circle x y ∧ movingCircle cx cy r ∧ (x - cx)^2 + (y - cy)^2 = r^2

-- Define the trajectory of the center of the moving circle
def trajectory (x y : ℝ) : Prop :=
  ∃ (r : ℝ), isTangent x y r circle1 ∧ isTangent x y r circle2

-- Theorem statement
theorem trajectory_is_hyperbola :
  ∃ (a b : ℝ), ∀ (x y : ℝ), trajectory x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_hyperbola_l2077_207752


namespace NUMINAMATH_CALUDE_dried_fruit_percentage_l2077_207764

/-- Represents the composition of a trail mix -/
structure TrailMix where
  nuts : ℚ
  dried_fruit : ℚ
  chocolate_chips : ℚ
  composition_sum : nuts + dried_fruit + chocolate_chips = 1

/-- The combined mixture of two trail mixes -/
def combined_mixture (mix1 mix2 : TrailMix) : TrailMix :=
  { nuts := (mix1.nuts + mix2.nuts) / 2,
    dried_fruit := (mix1.dried_fruit + mix2.dried_fruit) / 2,
    chocolate_chips := (mix1.chocolate_chips + mix2.chocolate_chips) / 2,
    composition_sum := by sorry }

theorem dried_fruit_percentage
  (sue_mix : TrailMix)
  (jane_mix : TrailMix)
  (h_sue_nuts : sue_mix.nuts = 0.3)
  (h_sue_dried_fruit : sue_mix.dried_fruit = 0.7)
  (h_jane_nuts : jane_mix.nuts = 0.6)
  (h_jane_chocolate : jane_mix.chocolate_chips = 0.4)
  (h_combined_nuts : (combined_mixture sue_mix jane_mix).nuts = 0.45) :
  (combined_mixture sue_mix jane_mix).dried_fruit = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_dried_fruit_percentage_l2077_207764


namespace NUMINAMATH_CALUDE_equiv_class_characterization_l2077_207732

/-- Given a positive integer m and an integer a, this theorem states that 
    an integer b is in the equivalence class of a modulo m if and only if 
    there exists an integer t such that b = m * t + a. -/
theorem equiv_class_characterization (m : ℕ+) (a b : ℤ) : 
  b ≡ a [ZMOD m] ↔ ∃ t : ℤ, b = m * t + a := by sorry

end NUMINAMATH_CALUDE_equiv_class_characterization_l2077_207732


namespace NUMINAMATH_CALUDE_max_value_expression_l2077_207787

theorem max_value_expression (y : ℝ) (h : y > 0) :
  (y^2 + 3 - Real.sqrt (y^4 + 9)) / y ≤ 4 * Real.sqrt 6 - 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2077_207787


namespace NUMINAMATH_CALUDE_megan_problem_solving_rate_l2077_207729

theorem megan_problem_solving_rate 
  (math_problems : ℕ) 
  (spelling_problems : ℕ) 
  (total_hours : ℕ) 
  (h1 : math_problems = 36)
  (h2 : spelling_problems = 28)
  (h3 : total_hours = 8) :
  (math_problems + spelling_problems) / total_hours = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_megan_problem_solving_rate_l2077_207729


namespace NUMINAMATH_CALUDE_bottle_cap_cost_l2077_207725

/-- Given that 5 bottle caps cost $25, prove that each bottle cap costs $5. -/
theorem bottle_cap_cost : 
  ∀ (cost_per_cap : ℚ), 
  (5 : ℚ) * cost_per_cap = 25 → cost_per_cap = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_cost_l2077_207725


namespace NUMINAMATH_CALUDE_second_graders_borrowed_books_l2077_207724

theorem second_graders_borrowed_books (initial_books : ℕ) (remaining_books : ℕ) 
  (h1 : initial_books = 75) 
  (h2 : remaining_books = 57) : 
  initial_books - remaining_books = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_graders_borrowed_books_l2077_207724


namespace NUMINAMATH_CALUDE_cynthia_water_balloons_l2077_207738

/-- The number of water balloons each person has -/
structure WaterBalloons where
  janice : ℕ
  randy : ℕ
  cynthia : ℕ

/-- The conditions of the water balloon distribution -/
def water_balloon_conditions (wb : WaterBalloons) : Prop :=
  wb.janice = 6 ∧
  wb.randy = wb.janice / 2 ∧
  wb.cynthia = 4 * wb.randy

theorem cynthia_water_balloons (wb : WaterBalloons) 
  (h : water_balloon_conditions wb) : wb.cynthia = 12 := by
  sorry

#check cynthia_water_balloons

end NUMINAMATH_CALUDE_cynthia_water_balloons_l2077_207738


namespace NUMINAMATH_CALUDE_parallel_vectors_l2077_207756

theorem parallel_vectors (x : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ k • (1, x) = (x - 1, 2)) → x = 1 ∨ x = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2077_207756


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2077_207741

theorem pure_imaginary_complex_fraction (a : ℝ) :
  let z : ℂ := (a - Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2077_207741


namespace NUMINAMATH_CALUDE_gcd_of_324_243_270_l2077_207757

theorem gcd_of_324_243_270 : Nat.gcd 324 (Nat.gcd 243 270) = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_324_243_270_l2077_207757


namespace NUMINAMATH_CALUDE_blue_garden_yield_l2077_207769

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected carrot yield from a rectangular garden -/
def expectedCarrotYield (garden : GardenDimensions) (stepLength : ℝ) (yieldPerSqFt : ℝ) : ℝ :=
  (garden.length : ℝ) * stepLength * (garden.width : ℝ) * stepLength * yieldPerSqFt

/-- Theorem stating the expected carrot yield for Mr. Blue's garden -/
theorem blue_garden_yield :
  let garden : GardenDimensions := ⟨18, 25⟩
  let stepLength : ℝ := 3
  let yieldPerSqFt : ℝ := 3 / 4
  expectedCarrotYield garden stepLength yieldPerSqFt = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_blue_garden_yield_l2077_207769


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2077_207797

/-- Given vectors a and b in ℝ², prove that the cosine of the angle between them is √5 / 5 -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) : 
  a = (2, -4) → b = (-3, -4) → 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2077_207797


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2077_207776

/-- A regular dodecahedron -/
structure Dodecahedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 20
  edge_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of randomly selecting two vertices that are endpoints of an edge -/
def edge_endpoint_probability (d : Dodecahedron) : ℚ :=
  (d.edges.card : ℚ) / (d.vertices.card.choose 2 : ℚ)

/-- The main theorem -/
theorem dodecahedron_edge_probability (d : Dodecahedron) :
  edge_endpoint_probability d = 3/19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2077_207776


namespace NUMINAMATH_CALUDE_part_one_part_two_l2077_207744

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - (a + 1/a)*x + 1 < 0

def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Part 1
theorem part_one (x : ℝ) (h1 : p x 2) (h2 : q x) : 1 ≤ x ∧ x < 2 :=
sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, q x → p x a) (h_not_sufficient : ¬(∀ x, p x a → q x)) : 3 < a :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2077_207744


namespace NUMINAMATH_CALUDE_total_amount_proof_l2077_207775

/-- The total amount shared among p, q, and r -/
def total_amount : ℝ := 5400.000000000001

/-- The amount r has -/
def r_amount : ℝ := 3600.0000000000005

/-- Theorem stating that given r has two-thirds of the total amount and r's amount is 3600.0000000000005,
    the total amount is 5400.000000000001 -/
theorem total_amount_proof :
  (2 / 3 : ℝ) * total_amount = r_amount →
  total_amount = 5400.000000000001 := by
sorry

end NUMINAMATH_CALUDE_total_amount_proof_l2077_207775


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2077_207796

theorem necessary_but_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0) ∧
  (∃ x y : ℝ, x = 0 ∧ x^2 + y^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2077_207796


namespace NUMINAMATH_CALUDE_chessboard_constant_l2077_207761

/-- A function representing the numbers on an infinite chessboard. -/
def ChessboardFunction := ℤ × ℤ → ℝ

/-- The property that each number is the arithmetic mean of its four neighbors. -/
def IsMeanValue (f : ChessboardFunction) : Prop :=
  ∀ m n : ℤ, f (m, n) = (f (m+1, n) + f (m-1, n) + f (m, n+1) + f (m, n-1)) / 4

/-- The property that all values of the function are nonnegative. -/
def IsNonnegative (f : ChessboardFunction) : Prop :=
  ∀ m n : ℤ, 0 ≤ f (m, n)

/-- Theorem stating that a nonnegative function satisfying the mean value property is constant. -/
theorem chessboard_constant (f : ChessboardFunction) 
  (h_mean : IsMeanValue f) (h_nonneg : IsNonnegative f) : 
  ∃ c : ℝ, ∀ m n : ℤ, f (m, n) = c :=
sorry

end NUMINAMATH_CALUDE_chessboard_constant_l2077_207761


namespace NUMINAMATH_CALUDE_car_distance_ratio_l2077_207716

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (car : Car) : ℝ := car.speed * car.time

/-- Theorem: The ratio of distances covered by Car A and Car B is 3:1 -/
theorem car_distance_ratio :
  let car_a : Car := { speed := 50, time := 6 }
  let car_b : Car := { speed := 100, time := 1 }
  (distance car_a) / (distance car_b) = 3 := by
  sorry


end NUMINAMATH_CALUDE_car_distance_ratio_l2077_207716


namespace NUMINAMATH_CALUDE_triangle_theorem_l2077_207794

noncomputable def triangle_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  let S := (33 : ℝ) / 2
  3 * a = 5 * c * Real.sin A ∧
  Real.cos B = -(5 : ℝ) / 13 ∧
  S = (1 / 2) * a * c * Real.sin B →
  Real.sin A = (33 : ℝ) / 65 ∧
  b = 10

theorem triangle_theorem :
  ∀ (a b c : ℝ) (A B C : ℝ),
  triangle_proof a b c A B C :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2077_207794


namespace NUMINAMATH_CALUDE_xy_value_l2077_207721

theorem xy_value (x y : ℕ+) (h1 : x + y = 36) (h2 : 3 * x * y + 15 * x = 4 * y + 396) : x * y = 260 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2077_207721


namespace NUMINAMATH_CALUDE_simplify_cube_roots_product_l2077_207791

theorem simplify_cube_roots_product : 
  (1 + 27) ^ (1/3 : ℝ) * (1 + 27 ^ (1/3 : ℝ)) ^ (1/3 : ℝ) * (4 : ℝ) ^ (1/2 : ℝ) = 2 * 112 ^ (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_product_l2077_207791


namespace NUMINAMATH_CALUDE_halloween_candy_count_l2077_207767

/-- Calculates Haley's final candy count after Halloween -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Theorem: Given Haley's initial candy count, the amount she ate, and the amount she received,
    her final candy count is equal to 35. -/
theorem halloween_candy_count :
  final_candy_count 33 17 19 = 35 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l2077_207767


namespace NUMINAMATH_CALUDE_manuscript_completion_time_l2077_207766

/-- The time needed to complete the manuscript when two people work together after one has worked alone for some time. -/
theorem manuscript_completion_time
  (time_A : ℝ) -- Time for person A to complete the manuscript alone
  (time_B : ℝ) -- Time for person B to complete the manuscript alone
  (solo_work : ℝ) -- Time person A works alone before B joins
  (h_A_positive : time_A > 0)
  (h_B_positive : time_B > 0)
  (h_solo_work : 0 ≤ solo_work ∧ solo_work < time_A) :
  let remaining_time := (time_A * time_B - solo_work * time_B) / (time_A + time_B)
  remaining_time = 24 / 13 :=
by sorry

end NUMINAMATH_CALUDE_manuscript_completion_time_l2077_207766


namespace NUMINAMATH_CALUDE_excellent_set_properties_l2077_207745

-- Definition of an excellent set
def IsExcellentSet (M : Set ℝ) : Prop :=
  ∀ x y, x ∈ M → y ∈ M → (x + y) ∈ M ∧ (x - y) ∈ M

-- Theorem statement
theorem excellent_set_properties
  (A B : Set ℝ)
  (hA : IsExcellentSet A)
  (hB : IsExcellentSet B) :
  (IsExcellentSet (A ∩ B)) ∧
  (IsExcellentSet (A ∪ B) → (A ⊆ B ∨ B ⊆ A)) ∧
  (IsExcellentSet (A ∪ B) → IsExcellentSet (A ∩ B)) :=
by sorry

end NUMINAMATH_CALUDE_excellent_set_properties_l2077_207745
