import Mathlib

namespace NUMINAMATH_GPT_count_valid_N_l1510_151083

theorem count_valid_N : ∃ (count : ℕ), count = 10 ∧ 
    (∀ N : ℕ, (10 ≤ N ∧ N < 100) → 
        (∃ a b c d : ℕ, 
            a < 3 ∧ b < 3 ∧ c < 3 ∧ d < 4 ∧
            N = 3 * a + b ∧ N = 4 * c + d ∧
            2 * N % 50 = ((9 * a + b) + (8 * c + d)) % 50)) :=
sorry

end NUMINAMATH_GPT_count_valid_N_l1510_151083


namespace NUMINAMATH_GPT_garden_dimensions_l1510_151036

theorem garden_dimensions (w l : ℕ) (h₁ : l = w + 3) (h₂ : 2 * (l + w) = 26) : w = 5 ∧ l = 8 :=
by
  sorry

end NUMINAMATH_GPT_garden_dimensions_l1510_151036


namespace NUMINAMATH_GPT_sum_of_inserted_numbers_eq_12_l1510_151045

theorem sum_of_inserted_numbers_eq_12 (a b : ℝ) (r d : ℝ) 
  (h1 : a = 2 * r) 
  (h2 : b = 2 * r^2) 
  (h3 : b = a + d) 
  (h4 : 12 = b + d) : 
  a + b = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_inserted_numbers_eq_12_l1510_151045


namespace NUMINAMATH_GPT_solution_to_quadratic_inequality_l1510_151034

def quadratic_inequality (x : ℝ) : Prop := 3 * x^2 - 5 * x > 9

theorem solution_to_quadratic_inequality (x : ℝ) : quadratic_inequality x ↔ x < -1 ∨ x > 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_quadratic_inequality_l1510_151034


namespace NUMINAMATH_GPT_sum_of_digits_of_n_l1510_151094

theorem sum_of_digits_of_n : 
  ∃ n : ℕ, n > 1500 ∧ 
    (Nat.gcd 40 (n + 105) = 10) ∧ 
    (Nat.gcd (n + 40) 105 = 35) ∧ 
    (Nat.digits 10 n).sum = 8 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_n_l1510_151094


namespace NUMINAMATH_GPT_age_of_youngest_person_l1510_151055

theorem age_of_youngest_person :
  ∃ (a1 a2 a3 a4 : ℕ), 
  (a1 < a2) ∧ (a2 < a3) ∧ (a3 < a4) ∧ 
  (a4 = 50) ∧ 
  (a1 + a2 + a3 + a4 = 158) ∧ 
  (a2 - a1 = a3 - a2) ∧ (a3 - a2 = a4 - a3) ∧ 
  a1 = 29 :=
by
  sorry

end NUMINAMATH_GPT_age_of_youngest_person_l1510_151055


namespace NUMINAMATH_GPT_error_estimate_alternating_series_l1510_151086

theorem error_estimate_alternating_series :
  let S := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4) + (-(1 / 5)) 
  let S₄ := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4)
  ∃ ΔS : ℝ, ΔS = |-(1 / 5)| ∧ ΔS < 0.2 := by
  sorry

end NUMINAMATH_GPT_error_estimate_alternating_series_l1510_151086


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1510_151066

theorem boat_speed_in_still_water (V_b : ℝ) : 
    (∀ (stream_speed : ℝ) (travel_time : ℝ) (distance : ℝ), 
        stream_speed = 5 ∧ 
        travel_time = 5 ∧ 
        distance = 105 →
        distance = (V_b + stream_speed) * travel_time) → 
    V_b = 16 := 
by 
    intro h
    specialize h 5 5 105 
    have h1 : 105 = (V_b + 5) * 5 := h ⟨rfl, ⟨rfl, rfl⟩⟩
    sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1510_151066


namespace NUMINAMATH_GPT_correct_mark_l1510_151027

theorem correct_mark 
  (avg_wrong : ℝ := 60)
  (wrong_mark : ℝ := 90)
  (num_students : ℕ := 30)
  (avg_correct : ℝ := 57.5) :
  (wrong_mark - (avg_wrong * num_students - avg_correct * num_students)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_correct_mark_l1510_151027


namespace NUMINAMATH_GPT_total_books_l1510_151004

-- Conditions
def TimsBooks : Nat := 44
def SamsBooks : Nat := 52
def AlexsBooks : Nat := 65
def KatiesBooks : Nat := 37

-- Theorem Statement
theorem total_books :
  TimsBooks + SamsBooks + AlexsBooks + KatiesBooks = 198 :=
by
  sorry

end NUMINAMATH_GPT_total_books_l1510_151004


namespace NUMINAMATH_GPT_find_a_l1510_151003

noncomputable def lines_perpendicular (a : ℝ) (l1: ℝ × ℝ × ℝ) (l2: ℝ × ℝ × ℝ) : Prop :=
  let (A1, B1, C1) := l1
  let (A2, B2, C2) := l2
  (B1 ≠ 0) ∧ (B2 ≠ 0) ∧ (-A1 / B1) * (-A2 / B2) = -1

theorem find_a (a : ℝ) :
  lines_perpendicular a (a, 1, 1) (2*a, a - 3, 1) → a = 1 ∨ a = -3/2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1510_151003


namespace NUMINAMATH_GPT_problem_inequality_l1510_151015

theorem problem_inequality (a b c m n p : ℝ) (h1 : a + b + c = 1) (h2 : m + n + p = 1) :
  -1 ≤ a * m + b * n + c * p ∧ a * m + b * n + c * p ≤ 1 := by
  sorry

end NUMINAMATH_GPT_problem_inequality_l1510_151015


namespace NUMINAMATH_GPT_eq_of_operation_l1510_151037

theorem eq_of_operation {x : ℝ} (h : 60 + 5 * 12 / (x / 3) = 61) : x = 180 :=
by
  sorry

end NUMINAMATH_GPT_eq_of_operation_l1510_151037


namespace NUMINAMATH_GPT_problem_a_correct_answer_l1510_151076

def initial_digit_eq_six (n : ℕ) : Prop :=
∃ k a : ℕ, n = 6 * 10^k + a ∧ a = n / 25

theorem problem_a_correct_answer :
  ∀ n : ℕ, initial_digit_eq_six n ↔ ∃ m : ℕ, n = 625 * 10^m :=
by
  sorry

end NUMINAMATH_GPT_problem_a_correct_answer_l1510_151076


namespace NUMINAMATH_GPT_correct_statements_l1510_151019

namespace ProofProblem

def P1 : Prop := (-4) + (-5) = -9
def P2 : Prop := -5 - (-6) = 11
def P3 : Prop := -2 * (-10) = -20
def P4 : Prop := 4 / (-2) = -2

theorem correct_statements : P1 ∧ P4 ∧ ¬P2 ∧ ¬P3 := by
  -- proof to be filled in later
  sorry

end ProofProblem

end NUMINAMATH_GPT_correct_statements_l1510_151019


namespace NUMINAMATH_GPT_tan_identity_l1510_151060

theorem tan_identity :
  let t5 := Real.tan (Real.pi / 36) -- 5 degrees in radians
  let t40 := Real.tan (Real.pi / 9)  -- 40 degrees in radians
  t5 + t40 + t5 * t40 = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_identity_l1510_151060


namespace NUMINAMATH_GPT_select_monkey_l1510_151053

theorem select_monkey (consumption : ℕ → ℕ) (n bananas minutes : ℕ)
  (h1 : consumption 1 = 1) (h2 : consumption 2 = 2) (h3 : consumption 3 = 3)
  (h4 : consumption 4 = 4) (h5 : consumption 5 = 5) (h6 : consumption 6 = 6)
  (h_total_minutes : minutes = 18) (h_total_bananas : bananas = 18) :
  consumption 1 * minutes = bananas :=
by
  sorry

end NUMINAMATH_GPT_select_monkey_l1510_151053


namespace NUMINAMATH_GPT_pipe_network_renovation_l1510_151095

theorem pipe_network_renovation 
  (total_length : Real)
  (efficiency_increase : Real)
  (days_ahead_of_schedule : Nat)
  (days_completed : Nat)
  (total_period : Nat)
  (original_daily_renovation : Real)
  (additional_renovation : Real)
  (h1 : total_length = 3600)
  (h2 : efficiency_increase = 20 / 100)
  (h3 : days_ahead_of_schedule = 10)
  (h4 : days_completed = 20)
  (h5 : total_period = 40)
  (h6 : (3600 / original_daily_renovation) - (3600 / (1.2 * original_daily_renovation)) = 10)
  (h7 : 20 * (72 + additional_renovation) >= 3600 - 1440) :
  (1.2 * original_daily_renovation = 72) ∧ (additional_renovation >= 36) :=
by
  sorry

end NUMINAMATH_GPT_pipe_network_renovation_l1510_151095


namespace NUMINAMATH_GPT_eleven_percent_greater_than_seventy_l1510_151032

theorem eleven_percent_greater_than_seventy : ∀ x : ℝ, (x = 70 * (1 + 11 / 100)) → (x = 77.7) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_eleven_percent_greater_than_seventy_l1510_151032


namespace NUMINAMATH_GPT_lily_pads_half_lake_l1510_151074

theorem lily_pads_half_lake (n : ℕ) (h : n = 39) :
  (n - 1) = 38 :=
by
  sorry

end NUMINAMATH_GPT_lily_pads_half_lake_l1510_151074


namespace NUMINAMATH_GPT_triangle_fraction_correct_l1510_151011

def point : Type := ℤ × ℤ

def area_triangle (A B C : point) : ℚ :=
  (1 / 2 : ℚ) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℚ))

def area_grid (length width : ℚ) : ℚ :=
  length * width

noncomputable def fraction_covered (A B C : point) (grid_length grid_width : ℚ) : ℚ :=
  area_triangle A B C / area_grid grid_length grid_width

theorem triangle_fraction_correct :
  fraction_covered (-2, 3) (2, -2) (3, 5) 8 6 = 11 / 32 :=
by
  sorry

end NUMINAMATH_GPT_triangle_fraction_correct_l1510_151011


namespace NUMINAMATH_GPT_tom_paths_avoiding_construction_l1510_151030

def tom_home : (ℕ × ℕ) := (0, 0)
def friend_home : (ℕ × ℕ) := (4, 3)
def construction_site : (ℕ × ℕ) := (2, 2)

def total_paths_without_restriction : ℕ := Nat.choose 7 4
def paths_via_construction_site : ℕ := (Nat.choose 4 2) * (Nat.choose 3 1)
def valid_paths : ℕ := total_paths_without_restriction - paths_via_construction_site

theorem tom_paths_avoiding_construction : valid_paths = 17 := by
  sorry

end NUMINAMATH_GPT_tom_paths_avoiding_construction_l1510_151030


namespace NUMINAMATH_GPT_distance_per_trip_l1510_151014

--  Define the conditions as assumptions
variables (total_distance : ℝ) (num_trips : ℝ)
axiom h_total_distance : total_distance = 120
axiom h_num_trips : num_trips = 4

-- Define the question converted into a statement to be proven
theorem distance_per_trip : total_distance / num_trips = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_distance_per_trip_l1510_151014


namespace NUMINAMATH_GPT_find_m_l1510_151065

noncomputable def curve (x : ℝ) : ℝ := (1 / 4) * x^2
noncomputable def line (x : ℝ) : ℝ := 1 - 2 * x

theorem find_m (m n : ℝ) (h_curve : curve m = n) (h_perpendicular : (1 / 2) * m * (-2) = -1) : m = 1 := 
  sorry

end NUMINAMATH_GPT_find_m_l1510_151065


namespace NUMINAMATH_GPT_average_DE_l1510_151038

theorem average_DE 
  (a b c d e : ℝ) 
  (avg_all : (a + b + c + d + e) / 5 = 80) 
  (avg_abc : (a + b + c) / 3 = 78) : 
  (d + e) / 2 = 83 := 
sorry

end NUMINAMATH_GPT_average_DE_l1510_151038


namespace NUMINAMATH_GPT_false_statement_l1510_151080

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x

def p : Prop := ∃ x0 : ℝ, f x0 = -1
def q : Prop := ∀ x : ℝ, f (2 * Real.pi + x) = f x

theorem false_statement : ¬ (p ∧ q) := sorry

end NUMINAMATH_GPT_false_statement_l1510_151080


namespace NUMINAMATH_GPT_train_crossing_pole_time_l1510_151022

theorem train_crossing_pole_time :
  ∀ (speed_kmph length_m: ℝ), speed_kmph = 160 → length_m = 400.032 → 
  length_m / (speed_kmph * 1000 / 3600) = 9.00072 :=
by
  intros speed_kmph length_m h_speed h_length
  rw [h_speed, h_length]
  -- The proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_train_crossing_pole_time_l1510_151022


namespace NUMINAMATH_GPT_half_of_animals_get_sick_l1510_151097

theorem half_of_animals_get_sick : 
  let chickens := 26
  let piglets := 40
  let goats := 34
  let total_animals := chickens + piglets + goats
  let sick_animals := total_animals / 2
  sick_animals = 50 :=
by
  sorry

end NUMINAMATH_GPT_half_of_animals_get_sick_l1510_151097


namespace NUMINAMATH_GPT_combined_weight_l1510_151021

-- Definition of conditions
def regular_dinosaur_weight := 800
def five_regular_dinosaurs_weight := 5 * regular_dinosaur_weight
def barney_weight := five_regular_dinosaurs_weight + 1500

-- Statement to prove
theorem combined_weight (h1: five_regular_dinosaurs_weight = 5 * regular_dinosaur_weight)
                        (h2: barney_weight = five_regular_dinosaurs_weight + 1500) : 
        (barney_weight + five_regular_dinosaurs_weight = 9500) :=
by
    sorry

end NUMINAMATH_GPT_combined_weight_l1510_151021


namespace NUMINAMATH_GPT_divisible_by_55_l1510_151087

theorem divisible_by_55 (n : ℤ) : 
  (55 ∣ (n^2 + 3 * n + 1)) ↔ (n % 55 = 46 ∨ n % 55 = 6) := 
by 
  sorry

end NUMINAMATH_GPT_divisible_by_55_l1510_151087


namespace NUMINAMATH_GPT_martha_blue_butterflies_l1510_151068

variables (B Y : Nat)

theorem martha_blue_butterflies (h_total : B + Y + 5 = 11) (h_twice : B = 2 * Y) : B = 4 :=
by
  sorry

end NUMINAMATH_GPT_martha_blue_butterflies_l1510_151068


namespace NUMINAMATH_GPT_tax_on_other_items_l1510_151048

theorem tax_on_other_items (total_amount clothing_amount food_amount other_items_amount tax_on_clothing tax_on_food total_tax : ℝ) (tax_percent_other : ℝ) 
(h1 : clothing_amount = 0.5 * total_amount)
(h2 : food_amount = 0.2 * total_amount)
(h3 : other_items_amount = 0.3 * total_amount)
(h4 : tax_on_clothing = 0.04 * clothing_amount)
(h5 : tax_on_food = 0) 
(h6 : total_tax = 0.044 * total_amount)
: 
(tax_percent_other = 8) := 
by
  -- Definitions from the problem
  -- Define the total tax paid as the sum of taxes on clothing, food, and other items
  let tax_other_items : ℝ := tax_percent_other / 100 * other_items_amount
  
  -- Total tax equation
  have h7 : tax_on_clothing + tax_on_food + tax_other_items = total_tax
  sorry

  -- Substitution values into the given conditions and solving
  have h8 : tax_on_clothing + tax_percent_other / 100 * other_items_amount = total_tax
  sorry
  
  have h9 : 0.04 * 0.5 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h10 : 0.02 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h11 : tax_percent_other / 100 * 0.3 * total_amount = 0.024 * total_amount
  sorry

  have h12 : tax_percent_other / 100 * 0.3 = 0.024
  sorry

  have h13 : tax_percent_other / 100 = 0.08
  sorry

  have h14 : tax_percent_other = 8
  sorry

  exact h14

end NUMINAMATH_GPT_tax_on_other_items_l1510_151048


namespace NUMINAMATH_GPT_mark_exceeded_sugar_intake_by_100_percent_l1510_151000

-- Definitions of the conditions
def softDrinkCalories : ℕ := 2500
def sugarPercentage : ℝ := 0.05
def caloriesPerCandy : ℕ := 25
def numCandyBars : ℕ := 7
def recommendedSugarIntake : ℕ := 150

-- Calculating the amount of added sugar in the soft drink
def addedSugarSoftDrink : ℝ := sugarPercentage * softDrinkCalories

-- Calculating the total added sugar from the candy bars
def addedSugarCandyBars : ℕ := numCandyBars * caloriesPerCandy

-- Summing the added sugar from the soft drink and the candy bars
def totalAddedSugar : ℝ := addedSugarSoftDrink + (addedSugarCandyBars : ℝ)

-- Calculate the excess intake of added sugar over the recommended amount
def excessSugarIntake : ℝ := totalAddedSugar - (recommendedSugarIntake : ℝ)

-- Prove that the percentage by which Mark exceeded the recommended intake of added sugar is 100%
theorem mark_exceeded_sugar_intake_by_100_percent :
  (excessSugarIntake / (recommendedSugarIntake : ℝ)) * 100 = 100 :=
by
  sorry

end NUMINAMATH_GPT_mark_exceeded_sugar_intake_by_100_percent_l1510_151000


namespace NUMINAMATH_GPT_puzzle_solution_l1510_151090

theorem puzzle_solution :
  (∀ n m k : ℕ, n + m + k = 111 → 9 * (n + m + k) / 3 = 9) ∧
  (∀ n m k : ℕ, n + m + k = 444 → 12 * (n + m + k) / 12 = 12) ∧
  (∀ n m k : ℕ, n + m + k = 777 → (7 * 3 ≠ 15 → (7 * 3 - 6 = 15)) ) →
  ∀ n m k : ℕ, n + m + k = 888 → 8 * (n + m + k / 3) - 6 = 18 :=
by
  intros h n m k h1
  sorry

end NUMINAMATH_GPT_puzzle_solution_l1510_151090


namespace NUMINAMATH_GPT_points_earned_l1510_151023

-- Definition of the conditions explicitly stated in the problem
def points_per_bag := 8
def total_bags := 4
def bags_not_recycled := 2

-- Calculation of bags recycled
def bags_recycled := total_bags - bags_not_recycled

-- The main theorem stating the proof equivalent
theorem points_earned : points_per_bag * bags_recycled = 16 := 
by
  sorry

end NUMINAMATH_GPT_points_earned_l1510_151023


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l1510_151088

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 6.40)
  (h2 : (a + b) / 2 = 6.2)
  (h3 : (c + d) / 2 = 6.1) :
  ((e + f) / 2) = 6.9 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l1510_151088


namespace NUMINAMATH_GPT_oranges_in_bin_l1510_151078

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (added : ℕ) (result : ℕ)
    (h_initial : initial = 40)
    (h_thrown_away : thrown_away = 25)
    (h_added : added = 21)
    (h_result : result = 36) : initial - thrown_away + added = result :=
by
  -- skipped proof steps
  exact sorry

end NUMINAMATH_GPT_oranges_in_bin_l1510_151078


namespace NUMINAMATH_GPT_polynomial_not_33_l1510_151070

theorem polynomial_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end NUMINAMATH_GPT_polynomial_not_33_l1510_151070


namespace NUMINAMATH_GPT_math_problem_l1510_151075

theorem math_problem 
  (x y : ℝ) 
  (h : x^2 + y^2 - x * y = 1) 
  : (-2 ≤ x + y) ∧ (x^2 + y^2 ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1510_151075


namespace NUMINAMATH_GPT_library_books_l1510_151077

theorem library_books (N x y : ℕ) (h1 : x = N / 17) (h2 : y = x + 2000)
    (h3 : y = (N - 2 * 2000) / 15 + (14 * (N - 2000) / 17)): 
  N = 544000 := 
sorry

end NUMINAMATH_GPT_library_books_l1510_151077


namespace NUMINAMATH_GPT_bill_profit_difference_l1510_151013

theorem bill_profit_difference 
  (SP : ℝ) 
  (hSP : SP = 1.10 * (SP / 1.10)) 
  (hSP_val : SP = 989.9999999999992) 
  (NP : ℝ) 
  (hNP : NP = 0.90 * (SP / 1.10)) 
  (NSP : ℝ) 
  (hNSP : NSP = 1.30 * NP) 
  : NSP - SP = 63.0000000000008 := 
by 
  sorry

end NUMINAMATH_GPT_bill_profit_difference_l1510_151013


namespace NUMINAMATH_GPT_siblings_are_Emma_and_Olivia_l1510_151073

structure Child where
  name : String
  eyeColor : String
  hairColor : String
  ageGroup : String

def Bella := Child.mk "Bella" "Green" "Red" "Older"
def Derek := Child.mk "Derek" "Gray" "Red" "Younger"
def Olivia := Child.mk "Olivia" "Green" "Brown" "Older"
def Lucas := Child.mk "Lucas" "Gray" "Brown" "Younger"
def Emma := Child.mk "Emma" "Green" "Red" "Older"
def Ryan := Child.mk "Ryan" "Gray" "Red" "Older"
def Sophia := Child.mk "Sophia" "Green" "Brown" "Younger"
def Ethan := Child.mk "Ethan" "Gray" "Brown" "Older"

def sharesCharacteristics (c1 c2 : Child) : Nat :=
  (if c1.eyeColor = c2.eyeColor then 1 else 0) +
  (if c1.hairColor = c2.hairColor then 1 else 0) +
  (if c1.ageGroup = c2.ageGroup then 1 else 0)

theorem siblings_are_Emma_and_Olivia :
  sharesCharacteristics Bella Emma ≥ 2 ∧
  sharesCharacteristics Bella Olivia ≥ 2 ∧
  (sharesCharacteristics Bella Derek < 2) ∧
  (sharesCharacteristics Bella Lucas < 2) ∧
  (sharesCharacteristics Bella Ryan < 2) ∧
  (sharesCharacteristics Bella Sophia < 2) ∧
  (sharesCharacteristics Bella Ethan < 2) :=
by
  sorry

end NUMINAMATH_GPT_siblings_are_Emma_and_Olivia_l1510_151073


namespace NUMINAMATH_GPT_negation_example_l1510_151035

theorem negation_example :
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 1 < 0 :=
sorry

end NUMINAMATH_GPT_negation_example_l1510_151035


namespace NUMINAMATH_GPT_total_items_left_in_store_l1510_151098

noncomputable def items_ordered : ℕ := 4458
noncomputable def items_sold : ℕ := 1561
noncomputable def items_in_storeroom : ℕ := 575

theorem total_items_left_in_store : 
  (items_ordered - items_sold) + items_in_storeroom = 3472 := 
by 
  sorry

end NUMINAMATH_GPT_total_items_left_in_store_l1510_151098


namespace NUMINAMATH_GPT_no_integer_pair_2006_l1510_151089

theorem no_integer_pair_2006 : ∀ (x y : ℤ), x^2 - y^2 ≠ 2006 := by
  sorry

end NUMINAMATH_GPT_no_integer_pair_2006_l1510_151089


namespace NUMINAMATH_GPT_value_of_star_l1510_151040

theorem value_of_star (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : (a + b) % 4 = 0) : a^2 + 2*a*b + b^2 = 64 :=
by
  sorry

end NUMINAMATH_GPT_value_of_star_l1510_151040


namespace NUMINAMATH_GPT_max_int_solution_of_inequality_system_l1510_151041

theorem max_int_solution_of_inequality_system :
  ∃ (x : ℤ), (∀ (y : ℤ), (3 * y - 1 < y + 1) ∧ (2 * (2 * y - 1) ≤ 5 * y + 1) → y ≤ x) ∧
             (3 * x - 1 < x + 1) ∧ (2 * (2 * x - 1) ≤ 5 * x + 1) ∧
             x = 0 :=
by
  sorry

end NUMINAMATH_GPT_max_int_solution_of_inequality_system_l1510_151041


namespace NUMINAMATH_GPT_find_transform_l1510_151081

structure Vector3D (α : Type) := (x y z : α)

def T (u : Vector3D ℝ) : Vector3D ℝ := sorry

axiom linearity (a b : ℝ) (u v : Vector3D ℝ) : T (Vector3D.mk (a * u.x + b * v.x) (a * u.y + b * v.y) (a * u.z + b * v.z)) = 
                      Vector3D.mk (a * (T u).x + b * (T v).x) (a * (T u).y + b * (T v).y) (a * (T u).z + b * (T v).z)

axiom cross_product (u v : Vector3D ℝ) : T (Vector3D.mk (u.y * v.z - u.z * v.y) (u.z * v.x - u.x * v.z) (u.x * v.y - u.y * v.x)) = 
                    (Vector3D.mk ((T u).y * (T v).z - (T u).z * (T v).y) ((T u).z * (T v).x - (T u).x * (T v).z) ((T u).x * (T v).y - (T u).y * (T v).x))

axiom transform1 : T (Vector3D.mk 3 3 7) = Vector3D.mk 2 (-4) 5
axiom transform2 : T (Vector3D.mk (-2) 5 4) = Vector3D.mk 6 1 0

theorem find_transform : T (Vector3D.mk 5 15 11) = Vector3D.mk a b c := sorry

end NUMINAMATH_GPT_find_transform_l1510_151081


namespace NUMINAMATH_GPT_find_x_values_l1510_151043

theorem find_x_values (
  x : ℝ
) (h₁ : x ≠ 0) (h₂ : x ≠ 1) (h₃ : x ≠ 2) :
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 4) ↔ 
  (x < (1 - Real.sqrt 17) / 2 ∨ (0 < x ∧ x < 1) ∨ (2 < x ∧ x < (1 + Real.sqrt 17) / 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l1510_151043


namespace NUMINAMATH_GPT_volume_of_intersection_of_two_perpendicular_cylinders_l1510_151006

theorem volume_of_intersection_of_two_perpendicular_cylinders (R : ℝ) : 
  ∃ V : ℝ, V = (16 / 3) * R^3 := 
sorry

end NUMINAMATH_GPT_volume_of_intersection_of_two_perpendicular_cylinders_l1510_151006


namespace NUMINAMATH_GPT_inequality_real_equation_positive_integers_solution_l1510_151051

-- Prove the inequality for real numbers a and b
theorem inequality_real (a b : ℝ) :
  (a^2 + 1) * (b^2 + 1) + 50 ≥ 2 * ((2 * a + 1) * (3 * b + 1)) :=
  sorry

-- Find all positive integers n and p such that the equation holds
theorem equation_positive_integers_solution :
  ∃ (n p : ℕ), 0 < n ∧ 0 < p ∧ (n^2 + 1) * (p^2 + 1) + 45 = 2 * ((2 * n + 1) * (3 * p + 1)) ∧ n = 2 ∧ p = 2 :=
  sorry

end NUMINAMATH_GPT_inequality_real_equation_positive_integers_solution_l1510_151051


namespace NUMINAMATH_GPT_minimum_b_value_l1510_151012

theorem minimum_b_value (k : ℕ) (x y z b : ℕ) (h1 : x = 3 * k) (h2 : y = 4 * k)
  (h3 : z = 7 * k) (h4 : y = 15 * b - 5) (h5 : ∀ n : ℕ, n = 4 * k + 5 → n % 15 = 0) : 
  b = 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_b_value_l1510_151012


namespace NUMINAMATH_GPT_power_sum_eq_l1510_151072

theorem power_sum_eq : (-2)^2011 + (-2)^2012 = 2^2011 := by
  sorry

end NUMINAMATH_GPT_power_sum_eq_l1510_151072


namespace NUMINAMATH_GPT_trigonometric_equation_solution_l1510_151010

theorem trigonometric_equation_solution (n : ℕ) (h_pos : 0 < n) (x : ℝ) (hx1 : ∀ k : ℤ, x ≠ k * π / 2) :
  (1 / (Real.sin x)^(2 * n) + 1 / (Real.cos x)^(2 * n) = 2^(n + 1)) ↔ ∃ k : ℤ, x = (2 * k + 1) * π / 4 :=
by sorry

end NUMINAMATH_GPT_trigonometric_equation_solution_l1510_151010


namespace NUMINAMATH_GPT_line_through_two_points_l1510_151039

theorem line_through_two_points (x y : ℝ) (hA : (x, y) = (3, 0)) (hB : (x, y) = (0, 2)) :
  2 * x + 3 * y - 6 = 0 :=
sorry 

end NUMINAMATH_GPT_line_through_two_points_l1510_151039


namespace NUMINAMATH_GPT_max_sum_of_lengths_l1510_151024

def length_of_integer (k : ℤ) (hk : k > 1) : ℤ := sorry

theorem max_sum_of_lengths (x y : ℤ) (hx : x > 1) (hy : y > 1) (h : x + 3 * y < 920) :
  length_of_integer x hx + length_of_integer y hy = 15 :=
sorry

end NUMINAMATH_GPT_max_sum_of_lengths_l1510_151024


namespace NUMINAMATH_GPT_cost_of_one_stamp_l1510_151047

-- Defining the conditions
def cost_of_four_stamps := 136
def number_of_stamps := 4

-- Prove that if 4 stamps cost 136 cents, then one stamp costs 34 cents
theorem cost_of_one_stamp : cost_of_four_stamps / number_of_stamps = 34 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_stamp_l1510_151047


namespace NUMINAMATH_GPT_minimum_value_h_at_a_eq_2_range_of_a_l1510_151061

noncomputable def f (a x : ℝ) : ℝ := a * x + (a - 1) / x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x

theorem minimum_value_h_at_a_eq_2 : ∃ x, h 2 x = 3 := 
sorry

theorem range_of_a (a : ℝ) : (∀ x ≥ 1, h a x ≥ 1) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_h_at_a_eq_2_range_of_a_l1510_151061


namespace NUMINAMATH_GPT_cos_theta_is_correct_l1510_151009

def vector_1 : ℝ × ℝ := (4, 5)
def vector_2 : ℝ × ℝ := (2, 7)

noncomputable def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1 * v1.1 + v1.2 * v1.2) * Real.sqrt (v2.1 * v2.1 + v2.2 * v2.2))

theorem cos_theta_is_correct :
  cos_theta vector_1 vector_2 = 43 / (Real.sqrt 41 * Real.sqrt 53) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cos_theta_is_correct_l1510_151009


namespace NUMINAMATH_GPT_atomic_weight_of_chlorine_l1510_151046

theorem atomic_weight_of_chlorine (molecular_weight_AlCl3 : ℝ) (atomic_weight_Al : ℝ) (atomic_weight_Cl : ℝ) :
  molecular_weight_AlCl3 = 132 ∧ atomic_weight_Al = 26.98 →
  132 = 26.98 + 3 * atomic_weight_Cl →
  atomic_weight_Cl = 35.007 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_atomic_weight_of_chlorine_l1510_151046


namespace NUMINAMATH_GPT_determine_a_l1510_151042

theorem determine_a (a : ℝ) (x1 x2 : ℝ) :
  (x1 * x1 + (2 * a - 1) * x1 + a * a = 0) ∧
  (x2 * x2 + (2 * a - 1) * x2 + a * a = 0) ∧
  ((x1 + 2) * (x2 + 2) = 11) →
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l1510_151042


namespace NUMINAMATH_GPT_parallel_lines_slope_eq_l1510_151050

theorem parallel_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 ↔ 6 * x + m * y + 11 = 0) → m = 8 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_eq_l1510_151050


namespace NUMINAMATH_GPT_chessboard_no_single_black_square_l1510_151007

theorem chessboard_no_single_black_square :
  (∀ (repaint : (Fin 8) × Bool → (Fin 8) × Bool), False) :=
by 
  sorry

end NUMINAMATH_GPT_chessboard_no_single_black_square_l1510_151007


namespace NUMINAMATH_GPT_total_population_of_towns_l1510_151058

theorem total_population_of_towns :
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  num_towns * estimated_avg_pop = 95000 :=
by
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  show num_towns * estimated_avg_pop = 95000
  sorry

end NUMINAMATH_GPT_total_population_of_towns_l1510_151058


namespace NUMINAMATH_GPT_domain_of_f_l1510_151025

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 5)

theorem domain_of_f :
  { x : ℝ | f x ≠ 0 } = { x : ℝ | x ≠ -5 }
:= sorry

end NUMINAMATH_GPT_domain_of_f_l1510_151025


namespace NUMINAMATH_GPT_find_a_and_a100_l1510_151002

def seq (a : ℝ) (n : ℕ) : ℝ := (-1)^n * n + a

theorem find_a_and_a100 :
  ∃ a : ℝ, (seq a 1 + seq a 4 = 3 * seq a 2) ∧ (seq a 100 = 97) :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_a100_l1510_151002


namespace NUMINAMATH_GPT_distance_to_origin_eq_three_l1510_151026

theorem distance_to_origin_eq_three :
  let P := (1, 2, 2)
  let origin := (0, 0, 0)
  dist P origin = 3 := by
  sorry

end NUMINAMATH_GPT_distance_to_origin_eq_three_l1510_151026


namespace NUMINAMATH_GPT_length_of_intersection_segment_l1510_151005

-- Define the polar coordinates conditions
def curve_1 (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def curve_2 (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 1

-- Convert polar equations to Cartesian coordinates
def curve_1_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4 * y
def curve_2_cartesian (x y : ℝ) : Prop := x = 1

-- Define the intersection points and the segment length function
def segment_length (y1 y2 : ℝ) : ℝ := abs (y1 - y2)

-- The statement to prove
theorem length_of_intersection_segment :
  (curve_1_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_1_cartesian 1 (2 - Real.sqrt 3)) →
  (curve_2_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_2_cartesian 1 (2 - Real.sqrt 3)) →
  segment_length (2 + Real.sqrt 3) (2 - Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_length_of_intersection_segment_l1510_151005


namespace NUMINAMATH_GPT_sqrt_square_sub_sqrt2_l1510_151069

theorem sqrt_square_sub_sqrt2 (h : 1 < Real.sqrt 2) : Real.sqrt ((1 - Real.sqrt 2) ^ 2) = Real.sqrt 2 - 1 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_square_sub_sqrt2_l1510_151069


namespace NUMINAMATH_GPT_sequence_5th_term_l1510_151008

theorem sequence_5th_term (a b c : ℚ) (h1 : a = 1 / 4 * (4 + b)) (h2 : b = 1 / 4 * (a + 40)) (h3 : 40 = 1 / 4 * (b + c)) : 
  c = 2236 / 15 := 
by 
  sorry

end NUMINAMATH_GPT_sequence_5th_term_l1510_151008


namespace NUMINAMATH_GPT_solution_set_inequality_l1510_151052

theorem solution_set_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (x - a) * (x - (1 / a)) < 0} = {x : ℝ | a < x ∧ x < 1 / a} := sorry

end NUMINAMATH_GPT_solution_set_inequality_l1510_151052


namespace NUMINAMATH_GPT_rhombus_area_600_l1510_151017

noncomputable def area_of_rhombus (x y : ℝ) : ℝ := (x * y) * 2

theorem rhombus_area_600 (x y : ℝ) (qx qy : ℝ)
  (hx : x = 15) (hy : y = 20)
  (hr1 : qx = 15) (hr2 : qy = 20)
  (h_ratio : qy / qx = 4 / 3) :
  area_of_rhombus (2 * (x + y - 2)) (x + y) = 600 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_rhombus_area_600_l1510_151017


namespace NUMINAMATH_GPT_jennifer_fruits_left_l1510_151057

theorem jennifer_fruits_left:
  (apples = 2 * pears) →
  (cherries = oranges / 2) →
  (grapes = 3 * apples) →
  pears = 15 →
  oranges = 30 →
  pears_given = 3 →
  oranges_given = 5 →
  apples_given = 5 →
  cherries_given = 7 →
  grapes_given = 3 →
  (remaining_fruits =
    (pears - pears_given) +
    (oranges - oranges_given) +
    (apples - apples_given) +
    (cherries - cherries_given) +
    (grapes - grapes_given)) →
  remaining_fruits = 157 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jennifer_fruits_left_l1510_151057


namespace NUMINAMATH_GPT_find_c_l1510_151029

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ x ∈ Set.Ioo (-(7 / 3) : ℝ) (2 : ℝ)) → c = 14 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_c_l1510_151029


namespace NUMINAMATH_GPT_solve_for_c_l1510_151085

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem solve_for_c (a b c d : ℝ) 
    (h : ∀ x : ℝ, quadratic_function a b c x ≥ d) : c = d + b^2 / (4 * a) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_c_l1510_151085


namespace NUMINAMATH_GPT_parabola_directrix_l1510_151062

theorem parabola_directrix (p : ℝ) (hp : p > 0) 
  (x1 x2 t : ℝ) 
  (h_intersect : ∃ y1 y2, y1 = x1 + t ∧ y2 = x2 + t ∧ x1^2 = 2 * p * y1 ∧ x2^2 = 2 * p * y2)
  (h_midpoint : (x1 + x2) / 2 = 2) :
  p = 2 → ∃ d : ℝ, d = -1 := 
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1510_151062


namespace NUMINAMATH_GPT_find_a_plus_b_l1510_151071

theorem find_a_plus_b (a b : ℕ) (positive_a : 0 < a) (positive_b : 0 < b)
  (condition : ∀ (n : ℕ), (n > 0) → (∃ m n : ℕ, n = m * a + n * b) ∨ (∃ k l : ℕ, n = 2009 + k * a + l * b))
  (not_expressible : ∃ m n : ℕ, 1776 = m * a + n * b): a + b = 133 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l1510_151071


namespace NUMINAMATH_GPT_triangle_inequality_l1510_151091

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  3 * (a * b + a * c + b * c) ≤ (a + b + c) ^ 2 ∧ (a + b + c) ^ 2 < 4 * (a * b + a * c + b * c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1510_151091


namespace NUMINAMATH_GPT_probability_three_even_l1510_151099

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the probability of exactly three dice showing an even number
noncomputable def prob_exactly_three_even (n : ℕ) (k : ℕ) (p : ℚ) : ℚ := 
  (binomial n k : ℚ) * (p^k) * ((1 - p)^(n - k))

-- The main theorem stating the desired probability
theorem probability_three_even (n : ℕ) (p : ℚ) (k : ℕ) (h₁ : n = 6) (h₂ : p = 1/2) (h₃ : k = 3) :
  prob_exactly_three_even n k p = 5 / 16 := by
  sorry

-- Include required definitions and expected values for the theorem
#check binomial
#check prob_exactly_three_even
#check probability_three_even

end NUMINAMATH_GPT_probability_three_even_l1510_151099


namespace NUMINAMATH_GPT_original_wattage_l1510_151001

theorem original_wattage (W : ℝ) (new_W : ℝ) (h1 : new_W = 1.25 * W) (h2 : new_W = 100) : W = 80 :=
by
  sorry

end NUMINAMATH_GPT_original_wattage_l1510_151001


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l1510_151020

-- Define the conditions p and q
def p (x : ℝ) := x^2 < 5 * x - 6
def q (x : ℝ) := |x + 1| ≤ 4

-- The goal to prove
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) :=
by 
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l1510_151020


namespace NUMINAMATH_GPT_flour_per_special_crust_l1510_151079

-- Definitions of daily pie crusts and flour usage for standard crusts
def daily_pie_crusts := 50
def flour_per_standard_crust := 1 / 10
def total_daily_flour := daily_pie_crusts * flour_per_standard_crust

-- Definitions for special pie crusts today
def special_pie_crusts := 25
def total_special_flour := total_daily_flour / special_pie_crusts

-- Problem statement in Lean
theorem flour_per_special_crust :
  total_special_flour = 1 / 5 := by
  sorry

end NUMINAMATH_GPT_flour_per_special_crust_l1510_151079


namespace NUMINAMATH_GPT_probability_of_stopping_on_H_l1510_151092

theorem probability_of_stopping_on_H (y : ℚ)
  (h1 : (1 / 5) + (1 / 4) + y + y + (1 / 10) = 1)
  : y = 9 / 40 :=
sorry

end NUMINAMATH_GPT_probability_of_stopping_on_H_l1510_151092


namespace NUMINAMATH_GPT_sum_digits_2_2005_times_5_2007_times_3_l1510_151054

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem sum_digits_2_2005_times_5_2007_times_3 : 
  sum_of_digits (2^2005 * 5^2007 * 3) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_sum_digits_2_2005_times_5_2007_times_3_l1510_151054


namespace NUMINAMATH_GPT_f_zero_f_odd_solve_inequality_l1510_151059

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom increasing_on_nonneg : ∀ {x y : ℝ}, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem f_zero : f 0 = 0 :=
by sorry

theorem f_odd (x : ℝ) : f (-x) = -f x :=
by sorry

theorem solve_inequality {x : ℝ} (h : 0 < x) : f (Real.log x / Real.log 10 - 1) < 0 ↔ 0 < x ∧ x < 10 :=
by sorry

end NUMINAMATH_GPT_f_zero_f_odd_solve_inequality_l1510_151059


namespace NUMINAMATH_GPT_sum_first_9_terms_l1510_151093

variable (a b : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∀ m n k l, m + n = k + l → a m * a n = a k * a l
def geometric_prop (a : ℕ → ℝ) : Prop := a 3 * a 7 = 2 * a 5
def arithmetic_b5_eq_a5 (a b : ℕ → ℝ) : Prop := b 5 = a 5

-- The Sum Sn of an arithmetic sequence up to the nth terms
def arithmetic_sum (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (b 1 + b n)

-- Question statement: proving the required sum
theorem sum_first_9_terms (a b : ℕ → ℝ) (S : ℕ → ℝ) 
  (hg : is_geometric_sequence a) 
  (hp : geometric_prop a) 
  (hb : arithmetic_b5_eq_a5 a b) 
  (arith_sum: arithmetic_sum b S) :
  S 9 = 18 :=
  sorry

end NUMINAMATH_GPT_sum_first_9_terms_l1510_151093


namespace NUMINAMATH_GPT_tileability_condition_l1510_151084

theorem tileability_condition (a b k m n : ℕ) (h₁ : k ∣ a) (h₂ : k ∣ b) (h₃ : ∃ (t : Nat), t * (a * b) = m * n) : 
  2 * k ∣ m ∨ 2 * k ∣ n := 
sorry

end NUMINAMATH_GPT_tileability_condition_l1510_151084


namespace NUMINAMATH_GPT_MrFletcher_paid_l1510_151028

noncomputable def total_payment (hours_day1 hours_day2 hours_day3 rate_per_hour men : ℕ) : ℕ :=
  let total_hours := hours_day1 + hours_day2 + hours_day3
  let total_man_hours := total_hours * men
  total_man_hours * rate_per_hour

theorem MrFletcher_paid
  (hours_day1 hours_day2 hours_day3 : ℕ)
  (rate_per_hour men : ℕ)
  (h1 : hours_day1 = 10)
  (h2 : hours_day2 = 8)
  (h3 : hours_day3 = 15)
  (h4 : rate_per_hour = 10)
  (h5 : men = 2) :
  total_payment hours_day1 hours_day2 hours_day3 rate_per_hour men = 660 := 
by {
  -- skipped proof details
  sorry
}

end NUMINAMATH_GPT_MrFletcher_paid_l1510_151028


namespace NUMINAMATH_GPT_third_offense_fraction_l1510_151018

-- Define the conditions
def sentence_assault : ℕ := 3
def sentence_poisoning : ℕ := 24
def total_sentence : ℕ := 36

-- The main theorem to prove
theorem third_offense_fraction :
  (total_sentence - (sentence_assault + sentence_poisoning)) / (sentence_assault + sentence_poisoning) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_third_offense_fraction_l1510_151018


namespace NUMINAMATH_GPT_sum_zero_of_cubic_identity_l1510_151044

theorem sum_zero_of_cubic_identity (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a^3 + b^3 + c^3 = 3 * a * b * c) : 
  a + b + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_zero_of_cubic_identity_l1510_151044


namespace NUMINAMATH_GPT_upper_limit_of_raise_l1510_151063

theorem upper_limit_of_raise (lower upper : ℝ) (h_lower : lower = 0.05)
  (h_upper : upper > 0.08) (h_inequality : ∀ r, lower < r → r < upper)
  : upper < 0.09 :=
sorry

end NUMINAMATH_GPT_upper_limit_of_raise_l1510_151063


namespace NUMINAMATH_GPT_molecular_weight_constant_l1510_151096

-- Given condition
def molecular_weight (compound : Type) : ℝ := 260

-- Proof problem statement (no proof yet)
theorem molecular_weight_constant (compound : Type) : molecular_weight compound = 260 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_constant_l1510_151096


namespace NUMINAMATH_GPT_bail_rate_l1510_151082

theorem bail_rate 
  (distance_to_shore : ℝ) 
  (shore_speed : ℝ) 
  (leak_rate : ℝ) 
  (boat_capacity : ℝ) 
  (time_to_shore_min : ℝ) 
  (net_water_intake : ℝ)
  (r : ℝ) :
  distance_to_shore = 2 →
  shore_speed = 3 →
  leak_rate = 12 →
  boat_capacity = 40 →
  time_to_shore_min = 40 →
  net_water_intake = leak_rate - r →
  net_water_intake * (time_to_shore_min) ≤ boat_capacity →
  r ≥ 11 :=
by
  intros h_dist h_speed h_leak h_cap h_time h_net h_ineq
  sorry

end NUMINAMATH_GPT_bail_rate_l1510_151082


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1510_151067

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) - 3

theorem solution_set_of_inequality :
  { x : ℝ | f x < 0 } = { x : ℝ | x < Real.log 3 / Real.log 2 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1510_151067


namespace NUMINAMATH_GPT_hayley_friends_l1510_151031

theorem hayley_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (h1 : total_stickers = 72) (h2 : stickers_per_friend = 8) : (total_stickers / stickers_per_friend) = 9 :=
by
  sorry

end NUMINAMATH_GPT_hayley_friends_l1510_151031


namespace NUMINAMATH_GPT_highway_length_l1510_151064

theorem highway_length 
  (speed_car1 speed_car2 : ℕ) (time : ℕ)
  (h_speed_car1 : speed_car1 = 54)
  (h_speed_car2 : speed_car2 = 57)
  (h_time : time = 3) : 
  speed_car1 * time + speed_car2 * time = 333 := by
  sorry

end NUMINAMATH_GPT_highway_length_l1510_151064


namespace NUMINAMATH_GPT_Kelly_initial_games_l1510_151049

-- Condition definitions
variable (give_away : ℕ) (left_over : ℕ)
variable (initial_games : ℕ)

-- Given conditions
axiom h1 : give_away = 15
axiom h2 : left_over = 35

-- Proof statement
theorem Kelly_initial_games : initial_games = give_away + left_over :=
sorry

end NUMINAMATH_GPT_Kelly_initial_games_l1510_151049


namespace NUMINAMATH_GPT_sum_of_first_15_terms_l1510_151056

theorem sum_of_first_15_terms (a : ℕ → ℝ) (r : ℝ)
    (h_geom : ∀ n, a (n + 1) = a n * r)
    (h1 : a 1 + a 2 + a 3 = 1)
    (h2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 +
   a 10 + a 11 + a 12 + a 13 + a 14 + a 15) = 11 :=
sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_l1510_151056


namespace NUMINAMATH_GPT_decompose_zero_l1510_151033

theorem decompose_zero (a : ℤ) : 0 = 0 * a := by
  sorry

end NUMINAMATH_GPT_decompose_zero_l1510_151033


namespace NUMINAMATH_GPT_angle_CDB_45_degrees_l1510_151016

theorem angle_CDB_45_degrees
  (α β γ δ : ℝ)
  (triangle_isosceles_right : α = β)
  (triangle_angle_BCD : γ = 90)
  (square_angle_DCE : δ = 90)
  (triangle_angle_ABC : α = β)
  (isosceles_triangle_angle : α + β + γ = 180)
  (isosceles_triangle_right : α = 45)
  (isosceles_triangle_sum : α + α + 90 = 180)
  (square_geometry : δ = 90) :
  γ + δ = 180 →  180 - (γ + α) = 45 :=
by
  sorry

end NUMINAMATH_GPT_angle_CDB_45_degrees_l1510_151016
