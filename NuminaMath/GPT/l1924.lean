import Mathlib

namespace NUMINAMATH_GPT_mrs_hilt_travel_distance_l1924_192444

theorem mrs_hilt_travel_distance :
  let distance_water_fountain := 30
  let distance_main_office := 50
  let distance_teacher_lounge := 35
  let trips_water_fountain := 4
  let trips_main_office := 2
  let trips_teacher_lounge := 3
  (distance_water_fountain * trips_water_fountain +
   distance_main_office * trips_main_office +
   distance_teacher_lounge * trips_teacher_lounge) = 325 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mrs_hilt_travel_distance_l1924_192444


namespace NUMINAMATH_GPT_arithmetic_sequence_formula_and_sum_l1924_192402

theorem arithmetic_sequence_formula_and_sum 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h0 : a 1 = 1) 
  (h1 : a 3 = -3)
  (hS : ∃ k, S k = -35):
  (∀ n, a n = 3 - 2 * n) ∧ (∃ k, S k = -35 ∧ k = 7) :=
by
  -- Given an arithmetic sequence where a_1 = 1 and a_3 = -3,
  -- prove that the general formula is a_n = 3 - 2n
  -- and the sum of the first k terms S_k = -35 implies k = 7
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_formula_and_sum_l1924_192402


namespace NUMINAMATH_GPT_number_of_diamonds_in_F10_l1924_192457

def sequence_of_figures (F : ℕ → ℕ) : Prop :=
  F 1 = 4 ∧
  (∀ n ≥ 2, F n = F (n-1) + 4 * (n + 2)) ∧
  F 3 = 28

theorem number_of_diamonds_in_F10 (F : ℕ → ℕ) (h : sequence_of_figures F) : F 10 = 336 :=
by
  sorry

end NUMINAMATH_GPT_number_of_diamonds_in_F10_l1924_192457


namespace NUMINAMATH_GPT_car_balanced_by_cubes_l1924_192489

variable (M Ball Cube : ℝ)

-- Conditions from the problem
axiom condition1 : M = Ball + 2 * Cube
axiom condition2 : M + Cube = 2 * Ball

-- Theorem to prove
theorem car_balanced_by_cubes : M = 5 * Cube := sorry

end NUMINAMATH_GPT_car_balanced_by_cubes_l1924_192489


namespace NUMINAMATH_GPT_wood_pieces_gathered_l1924_192442

theorem wood_pieces_gathered (sacks : ℕ) (pieces_per_sack : ℕ) (total_pieces : ℕ)
  (h1 : sacks = 4)
  (h2 : pieces_per_sack = 20)
  (h3 : total_pieces = sacks * pieces_per_sack) :
  total_pieces = 80 :=
by
  sorry

end NUMINAMATH_GPT_wood_pieces_gathered_l1924_192442


namespace NUMINAMATH_GPT_exterior_angle_BAC_l1924_192497

theorem exterior_angle_BAC 
    (interior_angle_nonagon : ℕ → ℚ) 
    (angle_CAD_angle_BAD : ℚ → ℚ → ℚ)
    (exterior_angle_formula : ℚ → ℚ) :
  (interior_angle_nonagon 9 = 140) ∧ 
  (angle_CAD_angle_BAD 90 140 = 230) ∧ 
  (exterior_angle_formula 230 = 130) := 
sorry

end NUMINAMATH_GPT_exterior_angle_BAC_l1924_192497


namespace NUMINAMATH_GPT_time_taken_to_cross_platform_l1924_192400

noncomputable def length_of_train : ℝ := 100 -- in meters
noncomputable def speed_of_train_km_hr : ℝ := 60 -- in km/hr
noncomputable def length_of_platform : ℝ := 150 -- in meters

noncomputable def speed_of_train_m_s := speed_of_train_km_hr * (1000 / 3600) -- converting km/hr to m/s
noncomputable def total_distance := length_of_train + length_of_platform
noncomputable def time_taken := total_distance / speed_of_train_m_s

theorem time_taken_to_cross_platform : abs (time_taken - 15) < 0.1 :=
by
  sorry

end NUMINAMATH_GPT_time_taken_to_cross_platform_l1924_192400


namespace NUMINAMATH_GPT_jake_has_peaches_l1924_192452

variable (Jake Steven Jill : Nat)

def given_conditions : Prop :=
  (Steven = 15) ∧ (Steven = Jill + 14) ∧ (Jake = Steven - 7)

theorem jake_has_peaches (h : given_conditions Jake Steven Jill) : Jake = 8 :=
by
  cases h with
  | intro hs1 hrest =>
      cases hrest with
      | intro hs2 hs3 =>
          sorry

end NUMINAMATH_GPT_jake_has_peaches_l1924_192452


namespace NUMINAMATH_GPT_original_cost_is_49_l1924_192466

-- Define the conditions as assumptions
def original_cost_of_jeans (x : ℝ) : Prop :=
  let discounted_price := x / 2
  let wednesday_price := discounted_price - 10
  wednesday_price = 14.5

-- The theorem to prove
theorem original_cost_is_49 :
  ∃ x : ℝ, original_cost_of_jeans x ∧ x = 49 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_is_49_l1924_192466


namespace NUMINAMATH_GPT_train_overtake_distance_l1924_192492

/--
 Train A leaves the station traveling at 30 miles per hour.
 Two hours later, Train B leaves the same station traveling in the same direction at 42 miles per hour.
 Prove that Train A is overtaken by Train B 210 miles from the station.
-/
theorem train_overtake_distance
    (speed_A : ℕ) (speed_B : ℕ) (delay_B : ℕ)
    (hA : speed_A = 30)
    (hB : speed_B = 42)
    (hDelay : delay_B = 2) :
    ∃ d : ℕ, d = 210 ∧ ∀ t : ℕ, (speed_B * t = (speed_A * t + speed_A * delay_B) → d = speed_B * t) :=
by
  sorry

end NUMINAMATH_GPT_train_overtake_distance_l1924_192492


namespace NUMINAMATH_GPT_final_temperature_is_58_32_l1924_192443

-- Initial temperature
def T₀ : ℝ := 40

-- Sequence of temperature adjustments
def T₁ : ℝ := 2 * T₀
def T₂ : ℝ := T₁ - 30
def T₃ : ℝ := T₂ * (1 - 0.30)
def T₄ : ℝ := T₃ + 24
def T₅ : ℝ := T₄ * (1 - 0.10)
def T₆ : ℝ := T₅ + 8
def T₇ : ℝ := T₆ * (1 + 0.20)
def T₈ : ℝ := T₇ - 15

-- Proof statement
theorem final_temperature_is_58_32 : T₈ = 58.32 :=
by sorry

end NUMINAMATH_GPT_final_temperature_is_58_32_l1924_192443


namespace NUMINAMATH_GPT_prev_geng_yin_year_2010_is_1950_l1924_192486

def heavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def earthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "You", "Xu", "Hai"]

def cycleLength : Nat := Nat.lcm 10 12

def prev_geng_yin_year (current_year : Nat) : Nat :=
  if cycleLength ≠ 0 then
    current_year - cycleLength
  else
    current_year -- This line is just to handle the case where LCM is incorrectly zero, which shouldn't happen practically.

theorem prev_geng_yin_year_2010_is_1950 : prev_geng_yin_year 2010 = 1950 := by
  sorry

end NUMINAMATH_GPT_prev_geng_yin_year_2010_is_1950_l1924_192486


namespace NUMINAMATH_GPT_evaluate_A_minus10_3_l1924_192496

def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem evaluate_A_minus10_3 : A (-10) 3 = 1320 := 
  sorry

end NUMINAMATH_GPT_evaluate_A_minus10_3_l1924_192496


namespace NUMINAMATH_GPT_ratio_is_one_half_l1924_192403

noncomputable def ratio_of_females_to_males (f m : ℕ) (avg_female_age avg_male_age avg_total_age : ℕ) : ℚ :=
  (f : ℚ) / (m : ℚ)

theorem ratio_is_one_half (f m : ℕ) (avg_female_age avg_male_age avg_total_age : ℕ)
  (h_female_age : avg_female_age = 45)
  (h_male_age : avg_male_age = 30)
  (h_total_age : avg_total_age = 35)
  (h_total_avg : (45 * f + 30 * m) / (f + m) = 35) :
  ratio_of_females_to_males f m avg_female_age avg_male_age avg_total_age = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_is_one_half_l1924_192403


namespace NUMINAMATH_GPT_equal_roots_a_l1924_192435

theorem equal_roots_a {a : ℕ} :
  (a * a - 4 * (a + 3) = 0) → a = 6 := 
sorry

end NUMINAMATH_GPT_equal_roots_a_l1924_192435


namespace NUMINAMATH_GPT_nth_equation_l1924_192439

theorem nth_equation (n : ℕ) : 
  n ≥ 1 → (∃ k, k = n + 1 ∧ (k^2 - n^2 - 1) / 2 = n) :=
by
  intros h
  use n + 1
  sorry

end NUMINAMATH_GPT_nth_equation_l1924_192439


namespace NUMINAMATH_GPT_female_students_count_l1924_192458

variable (F M : ℕ)

def numberOfMaleStudents (F : ℕ) : ℕ := 3 * F

def totalStudents (F M : ℕ) : Prop := F + M = 52

theorem female_students_count :
  totalStudents F (numberOfMaleStudents F) → F = 13 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_female_students_count_l1924_192458


namespace NUMINAMATH_GPT_number_of_cars_l1924_192447

theorem number_of_cars (total_wheels cars_bikes trash_can tricycle roller_skates : ℕ) 
  (h1 : cars_bikes = 2) 
  (h2 : trash_can = 2) 
  (h3 : tricycle = 3) 
  (h4 : roller_skates = 4) 
  (h5 : total_wheels = 25) 
  : (total_wheels - (cars_bikes * 2 + trash_can * 2 + tricycle * 3 + roller_skates * 4)) / 4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cars_l1924_192447


namespace NUMINAMATH_GPT_Faye_age_l1924_192415

theorem Faye_age (D E C F : ℕ) (h1 : D = E - 5) (h2 : E = C + 3) (h3 : F = C + 2) (hD : D = 18) : F = 22 :=
by
  sorry

end NUMINAMATH_GPT_Faye_age_l1924_192415


namespace NUMINAMATH_GPT_find_A_l1924_192422

theorem find_A (A : ℤ) (h : 10 + A = 15) : A = 5 := by
  sorry

end NUMINAMATH_GPT_find_A_l1924_192422


namespace NUMINAMATH_GPT_solve_parallelogram_l1924_192426

variables (x y : ℚ)

def condition1 : Prop := 6 * y - 2 = 12 * y - 10
def condition2 : Prop := 4 * x + 5 = 8 * x + 1

theorem solve_parallelogram : condition1 y → condition2 x → x + y = 7 / 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_solve_parallelogram_l1924_192426


namespace NUMINAMATH_GPT_tan_half_product_values_l1924_192449

theorem tan_half_product_values (a b : ℝ) (h : 3 * (Real.sin a + Real.sin b) + 2 * (Real.sin a * Real.sin b + 1) = 0) : 
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = -4 ∨ x = -1) := sorry

end NUMINAMATH_GPT_tan_half_product_values_l1924_192449


namespace NUMINAMATH_GPT_value_of_2m_plus_3n_l1924_192414

theorem value_of_2m_plus_3n (m n : ℝ) (h : (m^2 + 4 * m + 5) * (n^2 - 2 * n + 6) = 5) : 2 * m + 3 * n = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_2m_plus_3n_l1924_192414


namespace NUMINAMATH_GPT_divisibility_of_n_pow_n_minus_1_l1924_192487

theorem divisibility_of_n_pow_n_minus_1 (n : ℕ) (h : n > 1): (n^ (n - 1) - 1) % (n - 1)^2 = 0 := 
  sorry

end NUMINAMATH_GPT_divisibility_of_n_pow_n_minus_1_l1924_192487


namespace NUMINAMATH_GPT_solution_is_singleton_l1924_192471

def solution_set : Set (ℝ × ℝ) := { (x, y) | 2 * x + y = 3 ∧ x - 2 * y = -1 }

theorem solution_is_singleton : solution_set = { (1, 1) } :=
by
  sorry

end NUMINAMATH_GPT_solution_is_singleton_l1924_192471


namespace NUMINAMATH_GPT_basketball_first_half_score_l1924_192433

/-- 
In a college basketball match between Team Alpha and Team Beta, the game was tied at the end 
of the second quarter. The number of points scored by Team Alpha in each of the four quarters
formed an increasing geometric sequence, and the number of points scored by Team Beta in each
of the four quarters formed an increasing arithmetic sequence. At the end of the fourth quarter, 
Team Alpha had won by two points, with neither team scoring more than 100 points. 
Prove that the total number of points scored by the two teams in the first half is 24.
-/
theorem basketball_first_half_score 
  (a r : ℕ) (b d : ℕ)
  (h1 : a + a * r = b + (b + d))
  (h2 : a + a * r + a * r^2 + a * r^3 = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
  (h3 : a + a * r + a * r^2 + a * r^3 ≤ 100)
  (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 100) : 
  a + a * r + b + (b + d) = 24 :=
  sorry

end NUMINAMATH_GPT_basketball_first_half_score_l1924_192433


namespace NUMINAMATH_GPT_find_valid_pairs_l1924_192404

def divides (a b : Nat) : Prop := ∃ k, b = a * k

def valid_pair (a b : Nat) : Prop :=
  divides (a^2 * b) (b^2 + 3 * a)

theorem find_valid_pairs :
  {ab | valid_pair ab.1 ab.2} = ({(1, 1), (1, 3)} : Set (Nat × Nat)) :=
by
  sorry

end NUMINAMATH_GPT_find_valid_pairs_l1924_192404


namespace NUMINAMATH_GPT_pentagon_product_condition_l1924_192475

theorem pentagon_product_condition :
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ a + b + c + d + e = 1 ∧
  ∃ (a' b' c' d' e' : ℝ), 
    (a', b', c', d', e') ∈ {perm | perm = (a, b, c, d, e) ∨ perm = (b, c, d, e, a) ∨ perm = (c, d, e, a, b) ∨ perm = (d, e, a, b, c) ∨ perm = (e, a, b, c, d)} ∧
    (a'*b' ≤ 1/9 ∧ b'*c' ≤ 1/9 ∧ c'*d' ≤ 1/9 ∧ d'*e' ≤ 1/9 ∧ e'*a' ≤ 1/9) := sorry

end NUMINAMATH_GPT_pentagon_product_condition_l1924_192475


namespace NUMINAMATH_GPT_price_increase_is_12_percent_l1924_192477

theorem price_increase_is_12_percent
    (P : ℝ) (d : ℝ) (P' : ℝ) (sale_price : ℝ) (increase : ℝ) (percentage_increase : ℝ) :
    P = 470 → d = 0.16 → P' = 442.18 → 
    sale_price = P - P * d →
    increase = P' - sale_price →
    percentage_increase = (increase / sale_price) * 100 →
    percentage_increase = 12 :=
  by
  sorry

end NUMINAMATH_GPT_price_increase_is_12_percent_l1924_192477


namespace NUMINAMATH_GPT_quadratic_form_sum_l1924_192484

theorem quadratic_form_sum :
  ∃ a b c : ℝ, (∀ x : ℝ, 5 * x^2 - 45 * x - 500 = a * (x + b)^2 + c) ∧ (a + b + c = -605.75) :=
sorry

end NUMINAMATH_GPT_quadratic_form_sum_l1924_192484


namespace NUMINAMATH_GPT_total_saltwater_animals_l1924_192401

variable (numSaltwaterAquariums : Nat)
variable (animalsPerAquarium : Nat)

theorem total_saltwater_animals (h1 : numSaltwaterAquariums = 22) (h2 : animalsPerAquarium = 46) : 
    numSaltwaterAquariums * animalsPerAquarium = 1012 := 
  by
    sorry

end NUMINAMATH_GPT_total_saltwater_animals_l1924_192401


namespace NUMINAMATH_GPT_rectangle_area_l1924_192425

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_rectangle_area_l1924_192425


namespace NUMINAMATH_GPT_find_pairs_of_positive_integers_l1924_192468

theorem find_pairs_of_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 = 4 * (x^2 * y + x * y^2 - 5) → (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_of_positive_integers_l1924_192468


namespace NUMINAMATH_GPT_common_difference_is_minus_two_l1924_192482

noncomputable def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
noncomputable def sum_arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem common_difference_is_minus_two
  (a1 d : ℤ)
  (h1 : sum_arith_seq a1 d 5 = 15)
  (h2 : arith_seq a1 d 2 = 5) :
  d = -2 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_is_minus_two_l1924_192482


namespace NUMINAMATH_GPT_exponent_division_example_l1924_192437

theorem exponent_division_example : ((3^2)^4) / (3^2) = 729 := by
  sorry

end NUMINAMATH_GPT_exponent_division_example_l1924_192437


namespace NUMINAMATH_GPT_find_f_at_2_l1924_192427

variable (f : ℝ → ℝ)
variable (k : ℝ)
variable (h1 : ∀ x, f x = x^3 + 3 * x * f'' 2)
variable (h2 : f' 2 = 12 + 3 * f' 2)

theorem find_f_at_2 : f 2 = -28 :=
by
  sorry

end NUMINAMATH_GPT_find_f_at_2_l1924_192427


namespace NUMINAMATH_GPT_total_ticket_cost_l1924_192440

theorem total_ticket_cost (adult_tickets student_tickets : ℕ) 
    (price_adult price_student : ℕ) 
    (total_tickets : ℕ) (n_adult_tickets : adult_tickets = 410) 
    (n_student_tickets : student_tickets = 436) 
    (p_adult : price_adult = 6) 
    (p_student : price_student = 3) 
    (total_tickets_sold : total_tickets = 846) : 
    (adult_tickets * price_adult + student_tickets * price_student) = 3768 :=
by
  sorry

end NUMINAMATH_GPT_total_ticket_cost_l1924_192440


namespace NUMINAMATH_GPT_measure_angle_ACB_l1924_192445

-- Definitions of angles and a given triangle
variable (α β γ : ℝ)
variable (angleABD angle75 : ℝ)
variable (triangleABC : Prop)

-- Conditions from the problem
def angle_supplementary : Prop := angleABD + α = 180
def sum_angles_triangle : Prop := α + β + γ = 180
def known_angle : Prop := β = 75
def angleABD_value : Prop := angleABD = 150

-- The theorem to prove
theorem measure_angle_ACB : 
  angle_supplementary angleABD α ∧
  sum_angles_triangle α β γ ∧
  known_angle β ∧
  angleABD_value angleABD
  → γ = 75 := by
  sorry


end NUMINAMATH_GPT_measure_angle_ACB_l1924_192445


namespace NUMINAMATH_GPT_B_cycling_speed_l1924_192438

variable (A_speed B_distance B_time B_speed : ℕ)
variable (t1 : ℕ := 7)
variable (d_total : ℕ := 140)
variable (B_catch_time : ℕ := 7)

theorem B_cycling_speed :
  A_speed = 10 → 
  d_total = 140 →
  B_catch_time = 7 → 
  B_speed = 20 :=
by
  sorry

end NUMINAMATH_GPT_B_cycling_speed_l1924_192438


namespace NUMINAMATH_GPT_find_rolls_of_toilet_paper_l1924_192491

theorem find_rolls_of_toilet_paper (visits : ℕ) (squares_per_visit : ℕ) (squares_per_roll : ℕ) (days : ℕ)
  (h_visits : visits = 3)
  (h_squares_per_visit : squares_per_visit = 5)
  (h_squares_per_roll : squares_per_roll = 300)
  (h_days : days = 20000) : (visits * squares_per_visit * days) / squares_per_roll = 1000 :=
by
  sorry

end NUMINAMATH_GPT_find_rolls_of_toilet_paper_l1924_192491


namespace NUMINAMATH_GPT_relationship_among_abc_l1924_192421

-- Define a, b, c
def a : ℕ := 22 ^ 55
def b : ℕ := 33 ^ 44
def c : ℕ := 55 ^ 33

-- State the theorem regarding the relationship among a, b, and c
theorem relationship_among_abc : a > b ∧ b > c := 
by
  -- Placeholder for the proof, not required for this task
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l1924_192421


namespace NUMINAMATH_GPT_range_of_a_l1924_192483

def condition1 (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def condition2 (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y

def exclusive_or (p q : Prop) : Prop :=
  (p ∧ ¬q) ∨ (¬p ∧ q)

theorem range_of_a (a : ℝ) :
  exclusive_or (condition1 a) (condition2 a) → (1 ≤ a ∧ a < 2) ∨ a ≤ -2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_range_of_a_l1924_192483


namespace NUMINAMATH_GPT_rational_sum_eq_neg2_l1924_192467

theorem rational_sum_eq_neg2 (a b : ℚ) (h : |a + 6| + (b - 4)^2 = 0) : a + b = -2 :=
sorry

end NUMINAMATH_GPT_rational_sum_eq_neg2_l1924_192467


namespace NUMINAMATH_GPT_equivalent_discount_l1924_192480

theorem equivalent_discount {x : ℝ} (h₀ : x > 0) :
    let first_discount := 0.10
    let second_discount := 0.20
    let single_discount := 0.28
    (1 - (1 - first_discount) * (1 - second_discount)) = single_discount := by
    sorry

end NUMINAMATH_GPT_equivalent_discount_l1924_192480


namespace NUMINAMATH_GPT_narrow_black_stripes_are_8_l1924_192478

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end NUMINAMATH_GPT_narrow_black_stripes_are_8_l1924_192478


namespace NUMINAMATH_GPT_condition_relationship_l1924_192456

noncomputable def M : Set ℝ := {x | x > 2}
noncomputable def P : Set ℝ := {x | x < 3}

theorem condition_relationship :
  ∀ x, (x ∈ (M ∩ P) → x ∈ (M ∪ P)) ∧ ¬ (x ∈ (M ∪ P) → x ∈ (M ∩ P)) :=
by
  sorry

end NUMINAMATH_GPT_condition_relationship_l1924_192456


namespace NUMINAMATH_GPT_no_twelve_consecutive_primes_in_ap_l1924_192420

theorem no_twelve_consecutive_primes_in_ap (d : ℕ) (h : d < 2000) :
  ∀ a : ℕ, ¬(∀ n : ℕ, n < 12 → (Prime (a + n * d))) :=
sorry

end NUMINAMATH_GPT_no_twelve_consecutive_primes_in_ap_l1924_192420


namespace NUMINAMATH_GPT_hexagon_area_is_20_l1924_192493

theorem hexagon_area_is_20 :
  let upper_base1 := 3
  let upper_base2 := 2
  let upper_height := 4
  let lower_base1 := 3
  let lower_base2 := 2
  let lower_height := 4
  let upper_trapezoid_area := (upper_base1 + upper_base2) * upper_height / 2
  let lower_trapezoid_area := (lower_base1 + lower_base2) * lower_height / 2
  let total_area := upper_trapezoid_area + lower_trapezoid_area
  total_area = 20 := 
by {
  sorry
}

end NUMINAMATH_GPT_hexagon_area_is_20_l1924_192493


namespace NUMINAMATH_GPT_ratio_length_width_l1924_192461

theorem ratio_length_width (A L W : ℕ) (hA : A = 432) (hW : W = 12) (hArea : A = L * W) : L / W = 3 := 
by
  -- Placeholders for the actual mathematical proof
  sorry

end NUMINAMATH_GPT_ratio_length_width_l1924_192461


namespace NUMINAMATH_GPT_floor_ceil_eq_l1924_192431

theorem floor_ceil_eq (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌊x⌋ - x = 0 :=
by
  sorry

end NUMINAMATH_GPT_floor_ceil_eq_l1924_192431


namespace NUMINAMATH_GPT_smallest_integer_n_satisfying_inequality_l1924_192408

theorem smallest_integer_n_satisfying_inequality :
  ∃ n : ℤ, n^2 - 13 * n + 36 ≤ 0 ∧ (∀ m : ℤ, m^2 - 13 * m + 36 ≤ 0 → m ≥ n) ∧ n = 4 := 
by
  sorry

end NUMINAMATH_GPT_smallest_integer_n_satisfying_inequality_l1924_192408


namespace NUMINAMATH_GPT_Nara_is_1_69_meters_l1924_192450

-- Define the heights of Sangheon, Chiho, and Nara
def Sangheon_height : ℝ := 1.56
def Chiho_height : ℝ := Sangheon_height - 0.14
def Nara_height : ℝ := Chiho_height + 0.27

-- The statement to be proven
theorem Nara_is_1_69_meters : Nara_height = 1.69 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_Nara_is_1_69_meters_l1924_192450


namespace NUMINAMATH_GPT_point_segment_length_eq_l1924_192454

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x ^ 2 / 25 + y ^ 2 / 16 = 1)

noncomputable def line_eq (x : ℝ) : Prop := (x = 3)

theorem point_segment_length_eq :
  ∀ (A B : ℝ × ℝ), (ellipse_eq A.1 A.2) → (ellipse_eq B.1 B.2) → 
  (line_eq A.1) → (line_eq B.1) → (A = (3, 16/5) ∨ A = (3, -16/5)) → 
  (B = (3, 16/5) ∨ B = (3, -16/5)) → 
  |A.2 - B.2| = 32 / 5 := sorry

end NUMINAMATH_GPT_point_segment_length_eq_l1924_192454


namespace NUMINAMATH_GPT_find_legs_of_right_triangle_l1924_192470

theorem find_legs_of_right_triangle (x y a Δ : ℝ) 
  (h1 : x^2 + y^2 = a^2) 
  (h2 : 2 * Δ = x * y) : 
  x = (Real.sqrt (a^2 + 4 * Δ) + Real.sqrt (a^2 - 4 * Δ)) / 2 ∧ 
  y = (Real.sqrt (a^2 + 4 * Δ) - Real.sqrt (a^2 - 4 * Δ)) / 2 :=
sorry

end NUMINAMATH_GPT_find_legs_of_right_triangle_l1924_192470


namespace NUMINAMATH_GPT_base5_to_base4_last_digit_l1924_192481

theorem base5_to_base4_last_digit (n : ℕ) (h : n = 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4) : (n % 4 = 2) :=
by sorry

end NUMINAMATH_GPT_base5_to_base4_last_digit_l1924_192481


namespace NUMINAMATH_GPT_cake_remaining_portion_l1924_192479

theorem cake_remaining_portion (initial_cake : ℝ) (alex_share_percentage : ℝ) (jordan_share_fraction : ℝ) :
  initial_cake = 1 ∧ alex_share_percentage = 0.4 ∧ jordan_share_fraction = 0.5 →
  (initial_cake - alex_share_percentage * initial_cake) * (1 - jordan_share_fraction) = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_cake_remaining_portion_l1924_192479


namespace NUMINAMATH_GPT_number_of_fifth_graders_l1924_192465

-- Define the conditions given in the problem.
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def buses : ℕ := 5
def seats_per_bus : ℕ := 72

-- Derived definitions with the help of the conditions.
def total_seats : ℕ := buses * seats_per_bus
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_sixth_and_seventh_graders : ℕ := sixth_graders + seventh_graders
def seats_taken : ℕ := total_sixth_and_seventh_graders + total_chaperones
def seats_for_fifth_graders : ℕ := total_seats - seats_taken

-- The final statement to prove the number of fifth graders.
theorem number_of_fifth_graders : seats_for_fifth_graders = 109 :=
by
  sorry

end NUMINAMATH_GPT_number_of_fifth_graders_l1924_192465


namespace NUMINAMATH_GPT_sum_distinct_prime_factors_of_n_l1924_192407

theorem sum_distinct_prime_factors_of_n (n : ℕ) 
    (h1 : n < 1000) 
    (h2 : ∃ k : ℕ, 42 * n = 180 * k) : 
    ∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ n % p1 = 0 ∧ n % p2 = 0 ∧ n % p3 = 0 ∧ p1 + p2 + p3 = 10 := 
sorry

end NUMINAMATH_GPT_sum_distinct_prime_factors_of_n_l1924_192407


namespace NUMINAMATH_GPT_medians_inequality_l1924_192409

  variable {a b c : ℝ} (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)

  noncomputable def median_length (a b c : ℝ) : ℝ :=
    1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2)

  noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
    (a + b + c) / 2

  theorem medians_inequality (m_a m_b m_c s: ℝ)
    (h_ma : m_a = median_length a b c)
    (h_mb : m_b = median_length b c a)
    (h_mc : m_c = median_length c a b)
    (h_s : s = semiperimeter a b c) :
    m_a^2 + m_b^2 + m_c^2 ≥ s^2 := by
  sorry
  
end NUMINAMATH_GPT_medians_inequality_l1924_192409


namespace NUMINAMATH_GPT_equation_of_circle_min_distance_PA_PB_l1924_192441

-- Definition of the given points, lines, and circle
def point (x y : ℝ) : Prop := true

def circle_through_points (x1 y1 x2 y2 x3 y3 : ℝ) (a b r : ℝ) : Prop :=
  (x1 + a) * (x1 + a) + y1 * y1 = r ∧
  (x2 + a) * (x2 + a) + y2 * y2 = r ∧
  (x3 + a) * (x3 + a) + y3 * y3 = r

def line (a b : ℝ) : Prop := true

-- Specific points
def D := point 0 1
def E := point (-2) 1
def F := point (-1) (Real.sqrt 2)

-- Lines l1 and l2
def l₁ (x : ℝ) : ℝ := x - 2
def l₂ (x : ℝ) : ℝ := x + 1

-- Intersection points A and B
def A := point 0 1
def B := point (-2) (-1)

-- Question Ⅰ: Find the equation of the circle
theorem equation_of_circle :
  ∃ a b r, circle_through_points 0 1 (-2) 1 (-1) (Real.sqrt 2) a b r ∧ (a = -1 ∧ b = 0 ∧ r = 2) :=
  sorry

-- Question Ⅱ: Find the minimum value of |PA|^2 + |PB|^2
def dist_sq (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

theorem min_distance_PA_PB :
  real := sorry

end NUMINAMATH_GPT_equation_of_circle_min_distance_PA_PB_l1924_192441


namespace NUMINAMATH_GPT_more_than_1000_triplets_l1924_192411

theorem more_than_1000_triplets :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 1000 < S.card ∧ 
  ∀ (a b c : ℕ), (a, b, c) ∈ S → a^15 + b^15 = c^16 :=
by sorry

end NUMINAMATH_GPT_more_than_1000_triplets_l1924_192411


namespace NUMINAMATH_GPT_product_of_real_roots_eq_one_l1924_192424

theorem product_of_real_roots_eq_one:
  ∀ x : ℝ, x ^ Real.log x = Real.exp 1 → (x = Real.exp 1 ∨ x = Real.exp (-1)) →
  x * (if x = Real.exp 1 then Real.exp (-1) else Real.exp 1) = 1 :=
by sorry

end NUMINAMATH_GPT_product_of_real_roots_eq_one_l1924_192424


namespace NUMINAMATH_GPT_hours_to_destination_l1924_192498

def num_people := 4
def water_per_person_per_hour := 1 / 2
def total_water_bottles_needed := 32

theorem hours_to_destination : 
  ∃ h : ℕ, (num_people * water_per_person_per_hour * 2 * h = total_water_bottles_needed) → h = 8 :=
by
  sorry

end NUMINAMATH_GPT_hours_to_destination_l1924_192498


namespace NUMINAMATH_GPT_aunt_li_more_cost_effective_l1924_192462

theorem aunt_li_more_cost_effective (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (100 * a + 100 * b) / 200 ≥ 200 / ((100 / a) + (100 / b)) :=
by
  sorry

end NUMINAMATH_GPT_aunt_li_more_cost_effective_l1924_192462


namespace NUMINAMATH_GPT_number_of_paths_grid_l1924_192499

def paths_from_A_to_C (h v : Nat) : Nat :=
  Nat.choose (h + v) v

#eval paths_from_A_to_C 7 6 -- expected result: 1716

theorem number_of_paths_grid :
  paths_from_A_to_C 7 6 = 1716 := by
  sorry

end NUMINAMATH_GPT_number_of_paths_grid_l1924_192499


namespace NUMINAMATH_GPT_jacob_has_more_money_l1924_192413

def exchange_rate : ℝ := 1.11
def Mrs_Hilt_total_in_dollars : ℝ := 
  3 * 0.01 + 2 * 0.10 + 2 * 0.05 + 5 * 0.25 + 1 * 1.00

def Jacob_total_in_euros : ℝ := 
  4 * 0.01 + 1 * 0.05 + 1 * 0.10 + 3 * 0.20 + 2 * 0.50 + 2 * 1.00

def Jacob_total_in_dollars : ℝ := Jacob_total_in_euros * exchange_rate

def difference : ℝ := Jacob_total_in_dollars - Mrs_Hilt_total_in_dollars

theorem jacob_has_more_money : difference = 1.63 :=
by sorry

end NUMINAMATH_GPT_jacob_has_more_money_l1924_192413


namespace NUMINAMATH_GPT_friends_meeting_both_movie_and_games_l1924_192451

theorem friends_meeting_both_movie_and_games 
  (T M P G M_and_P P_and_G M_and_P_and_G : ℕ) 
  (hT : T = 31) 
  (hM : M = 10) 
  (hP : P = 20) 
  (hG : G = 5) 
  (hM_and_P : M_and_P = 4) 
  (hP_and_G : P_and_G = 0) 
  (hM_and_P_and_G : M_and_P_and_G = 2) : (M + P + G - M_and_P - T + M_and_P_and_G - 2) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_friends_meeting_both_movie_and_games_l1924_192451


namespace NUMINAMATH_GPT_distance_between_points_l1924_192453

theorem distance_between_points :
  ∀ (D : ℝ), (10 + 2) * (5 / D) + (10 - 2) * (5 / D) = 24 ↔ D = 24 := 
sorry

end NUMINAMATH_GPT_distance_between_points_l1924_192453


namespace NUMINAMATH_GPT_binary_addition_subtraction_l1924_192473

def bin_10101 : ℕ := 0b10101
def bin_1011 : ℕ := 0b1011
def bin_1110 : ℕ := 0b1110
def bin_110001 : ℕ := 0b110001
def bin_1101 : ℕ := 0b1101
def bin_101100 : ℕ := 0b101100

theorem binary_addition_subtraction :
  bin_10101 + bin_1011 + bin_1110 + bin_110001 - bin_1101 = bin_101100 := 
sorry

end NUMINAMATH_GPT_binary_addition_subtraction_l1924_192473


namespace NUMINAMATH_GPT_abs_inequality_solution_l1924_192448

theorem abs_inequality_solution :
  { x : ℝ | |x - 2| + |x + 3| < 6 } = { x | -7 / 2 < x ∧ x < 5 / 2 } :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1924_192448


namespace NUMINAMATH_GPT_real_estate_commission_l1924_192464

theorem real_estate_commission (r : ℝ) (P : ℝ) (C : ℝ) (h : r = 0.06) (hp : P = 148000) : C = P * r :=
by
  -- Definitions and proof steps will go here.
  sorry

end NUMINAMATH_GPT_real_estate_commission_l1924_192464


namespace NUMINAMATH_GPT_min_value_of_squares_l1924_192428

theorem min_value_of_squares (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 ≥ 1990 :=
sorry

end NUMINAMATH_GPT_min_value_of_squares_l1924_192428


namespace NUMINAMATH_GPT_increase_by_150_percent_l1924_192476

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end NUMINAMATH_GPT_increase_by_150_percent_l1924_192476


namespace NUMINAMATH_GPT_extracurricular_popularity_order_l1924_192474

def fraction_likes_drama := 9 / 28
def fraction_likes_music := 13 / 36
def fraction_likes_art := 11 / 24

theorem extracurricular_popularity_order :
  fraction_likes_art > fraction_likes_music ∧ 
  fraction_likes_music > fraction_likes_drama :=
by
  sorry

end NUMINAMATH_GPT_extracurricular_popularity_order_l1924_192474


namespace NUMINAMATH_GPT_num_days_c_worked_l1924_192455

theorem num_days_c_worked (d : ℕ) :
  let daily_wage_c := 100
  let daily_wage_b := (4 * 20)
  let daily_wage_a := (3 * 20)
  let total_earning := 1480
  let earning_a := 6 * daily_wage_a
  let earning_b := 9 * daily_wage_b
  let earning_c := d * daily_wage_c
  total_earning = earning_a + earning_b + earning_c →
  d = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_num_days_c_worked_l1924_192455


namespace NUMINAMATH_GPT_max_servings_hot_chocolate_l1924_192488

def recipe_servings : ℕ := 5
def chocolate_required : ℕ := 2 -- squares of chocolate required for 5 servings
def sugar_required : ℚ := 1 / 4 -- cups of sugar required for 5 servings
def water_required : ℕ := 1 -- cups of water required (not limiting)
def milk_required : ℕ := 4 -- cups of milk required for 5 servings

def chocolate_available : ℕ := 5 -- squares of chocolate Jordan has
def sugar_available : ℚ := 2 -- cups of sugar Jordan has
def milk_available : ℕ := 7 -- cups of milk Jordan has
def water_available_lots : Prop := True -- Jordan has lots of water (not limited)

def servings_from_chocolate := (chocolate_available / chocolate_required) * recipe_servings
def servings_from_sugar := (sugar_available / sugar_required) * recipe_servings
def servings_from_milk := (milk_available / milk_required) * recipe_servings

def max_servings (a b c : ℚ) : ℚ := min (min a b) c

theorem max_servings_hot_chocolate :
  max_servings servings_from_chocolate servings_from_sugar servings_from_milk = 35 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_servings_hot_chocolate_l1924_192488


namespace NUMINAMATH_GPT_complex_division_l1924_192418

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (7 - i) / (3 + i) = 2 - i := by
  sorry

end NUMINAMATH_GPT_complex_division_l1924_192418


namespace NUMINAMATH_GPT_division_remainder_190_21_l1924_192434

theorem division_remainder_190_21 :
  190 = 21 * 9 + 1 :=
sorry

end NUMINAMATH_GPT_division_remainder_190_21_l1924_192434


namespace NUMINAMATH_GPT_sample_size_ratio_l1924_192472

theorem sample_size_ratio (n : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
                          (total_ratio : ℕ) (B_in_sample : ℕ)
                          (h_ratio : ratio_A = 1 ∧ ratio_B = 3 ∧ ratio_C = 5)
                          (h_total : total_ratio = ratio_A + ratio_B + ratio_C)
                          (h_B_sample : B_in_sample = 27)
                          (h_sampling_ratio_B : ratio_B / total_ratio = 1 / 3) :
                          n = 81 :=
by sorry

end NUMINAMATH_GPT_sample_size_ratio_l1924_192472


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1924_192412

theorem simplify_and_evaluate (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 2) (h₃ : x ≠ -2) :
  ((x - 1 - 3 / (x + 1)) / ((x^2 - 4) / (x^2 + 2 * x + 1))) = x + 1 ∧ ((x = 1) → (x + 1 = 2)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1924_192412


namespace NUMINAMATH_GPT_find_number_l1924_192406

theorem find_number (x : ℕ) : ((52 + x) * 3 - 60) / 8 = 15 → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1924_192406


namespace NUMINAMATH_GPT_hexagon_perimeter_l1924_192469

theorem hexagon_perimeter (side_length : ℝ) (sides : ℕ) (h_sides : sides = 6) (h_side_length : side_length = 10) :
  sides * side_length = 60 :=
by
  rw [h_sides, h_side_length]
  norm_num

end NUMINAMATH_GPT_hexagon_perimeter_l1924_192469


namespace NUMINAMATH_GPT_avg_of_combined_data_l1924_192430

variables (x1 x2 x3 y1 y2 y3 a b : ℝ)

-- condition: average of x1, x2, x3 is a
axiom h1 : (x1 + x2 + x3) / 3 = a

-- condition: average of y1, y2, y3 is b
axiom h2 : (y1 + y2 + y3) / 3 = b

-- Prove that the average of 3x1 + y1, 3x2 + y2, 3x3 + y3 is 3a + b
theorem avg_of_combined_data : 
  ((3 * x1 + y1) + (3 * x2 + y2) + (3 * x3 + y3)) / 3 = 3 * a + b :=
by
  sorry

end NUMINAMATH_GPT_avg_of_combined_data_l1924_192430


namespace NUMINAMATH_GPT_length_of_faster_train_is_370_l1924_192432

noncomputable def length_of_faster_train (vf vs : ℕ) (t : ℕ) : ℕ :=
  let rel_speed := vf - vs
  let rel_speed_m_per_s := rel_speed * 1000 / 3600
  rel_speed_m_per_s * t

theorem length_of_faster_train_is_370 :
  length_of_faster_train 72 36 37 = 370 := 
  sorry

end NUMINAMATH_GPT_length_of_faster_train_is_370_l1924_192432


namespace NUMINAMATH_GPT_stickers_total_proof_l1924_192436

def stickers_per_page : ℕ := 10
def number_of_pages : ℕ := 22
def total_stickers : ℕ := stickers_per_page * number_of_pages

theorem stickers_total_proof : total_stickers = 220 := by
  sorry

end NUMINAMATH_GPT_stickers_total_proof_l1924_192436


namespace NUMINAMATH_GPT_total_spent_after_three_years_l1924_192405

def iPhone_cost : ℝ := 1000
def contract_cost_per_month : ℝ := 200
def case_cost_before_discount : ℝ := 0.20 * iPhone_cost
def headphones_cost_before_discount : ℝ := 0.5 * case_cost_before_discount
def charger_cost : ℝ := 60
def warranty_cost_for_two_years : ℝ := 150
def discount_rate : ℝ := 0.10
def time_in_years : ℝ := 3

def contract_cost_for_three_years := contract_cost_per_month * 12 * time_in_years
def case_cost_after_discount := case_cost_before_discount * (1 - discount_rate)
def headphones_cost_after_discount := headphones_cost_before_discount * (1 - discount_rate)

def total_cost : ℝ :=
  iPhone_cost +
  contract_cost_for_three_years +
  case_cost_after_discount +
  headphones_cost_after_discount +
  charger_cost +
  warranty_cost_for_two_years

theorem total_spent_after_three_years : total_cost = 8680 :=
  by
    sorry

end NUMINAMATH_GPT_total_spent_after_three_years_l1924_192405


namespace NUMINAMATH_GPT_number_of_factors_l1924_192463

theorem number_of_factors (a b c d : ℕ) (h₁ : a = 6) (h₂ : b = 6) (h₃ : c = 5) (h₄ : d = 1) :
  ((a + 1) * (b + 1) * (c + 1) * (d + 1) = 588) :=
by {
  -- This is a placeholder for the actual proof
  sorry
}

end NUMINAMATH_GPT_number_of_factors_l1924_192463


namespace NUMINAMATH_GPT_milk_production_l1924_192459

-- Variables representing the problem parameters
variables {a b c f d e g : ℝ}

-- Preconditions
axiom pos_a : a > 0
axiom pos_c : c > 0
axiom pos_f : f > 0
axiom pos_d : d > 0
axiom pos_e : e > 0
axiom pos_g : g > 0

theorem milk_production (a b c f d e g : ℝ) (h_a : a > 0) (h_c : c > 0) (h_f : f > 0) (h_d : d > 0) (h_e : e > 0) (h_g : g > 0) :
  d * e * g * (b / (a * c * f)) = (b * d * e * g) / (a * c * f) := by
  sorry

end NUMINAMATH_GPT_milk_production_l1924_192459


namespace NUMINAMATH_GPT_size_of_each_group_l1924_192423

theorem size_of_each_group 
  (boys : ℕ) (girls : ℕ) (groups : ℕ)
  (total_students : boys + girls = 63)
  (num_groups : groups = 7) :
  63 / 7 = 9 :=
by
  sorry

end NUMINAMATH_GPT_size_of_each_group_l1924_192423


namespace NUMINAMATH_GPT_no_snuggly_two_digit_l1924_192490

theorem no_snuggly_two_digit (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) : ¬ (10 * a + b = a + b^3) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_snuggly_two_digit_l1924_192490


namespace NUMINAMATH_GPT_brownies_count_l1924_192460

theorem brownies_count {B : ℕ} 
  (h1 : B/2 = (B - B / 2))
  (h2 : B/4 = (B - B / 2) / 2)
  (h3 : B/4 - 2 = B/4 - 2)
  (h4 : B/4 - 2 = 3) : 
  B = 20 := 
by 
  sorry

end NUMINAMATH_GPT_brownies_count_l1924_192460


namespace NUMINAMATH_GPT_find_y_for_line_slope_45_degrees_l1924_192416

theorem find_y_for_line_slope_45_degrees :
  ∃ y, (∃ x₁ y₁ x₂ y₂, x₁ = 4 ∧ y₁ = y ∧ x₂ = 2 ∧ y₂ = -3 ∧ (y₂ - y₁) / (x₂ - x₁) = 1) → y = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_y_for_line_slope_45_degrees_l1924_192416


namespace NUMINAMATH_GPT_parallel_vectors_have_proportional_direction_ratios_l1924_192485

theorem parallel_vectors_have_proportional_direction_ratios (m : ℝ) :
  let a := (1, 2)
  let b := (m, 1)
  (a.1 / b.1) = (a.2 / b.2) → m = 1/2 :=
by
  let a := (1, 2)
  let b := (m, 1)
  intro h
  sorry

end NUMINAMATH_GPT_parallel_vectors_have_proportional_direction_ratios_l1924_192485


namespace NUMINAMATH_GPT_ratio_student_adult_tickets_l1924_192419

theorem ratio_student_adult_tickets (A : ℕ) (S : ℕ) (total_tickets: ℕ) (multiple: ℕ) :
  (A = 122) →
  (total_tickets = 366) →
  (S = multiple * A) →
  (S + A = total_tickets) →
  (S / A = 2) :=
by
  intros hA hTotal hMultiple hSum
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_ratio_student_adult_tickets_l1924_192419


namespace NUMINAMATH_GPT_flagpole_height_l1924_192494

theorem flagpole_height :
  ∃ (AB AC AD DE DC : ℝ), 
    AC = 5 ∧
    AD = 3 ∧ 
    DE = 1.8 ∧
    DC = AC - AD ∧
    AB = (DE * AC) / DC ∧
    AB = 4.5 :=
by
  exists 4.5, 5, 3, 1.8, 2
  simp
  sorry

end NUMINAMATH_GPT_flagpole_height_l1924_192494


namespace NUMINAMATH_GPT_chess_grandmaster_time_l1924_192495

theorem chess_grandmaster_time :
  let time_to_learn_rules : ℕ := 2
  let factor_to_get_proficient : ℕ := 49
  let factor_to_become_master : ℕ := 100
  let time_to_get_proficient := factor_to_get_proficient * time_to_learn_rules
  let combined_time := time_to_learn_rules + time_to_get_proficient
  let time_to_become_master := factor_to_become_master * combined_time
  let total_time := time_to_learn_rules + time_to_get_proficient + time_to_become_master
  total_time = 10100 :=
by
  sorry

end NUMINAMATH_GPT_chess_grandmaster_time_l1924_192495


namespace NUMINAMATH_GPT_ratio_of_boys_l1924_192446

theorem ratio_of_boys (p : ℝ) (h : p = (3/4) * (1 - p)) : 
  p = 3 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_boys_l1924_192446


namespace NUMINAMATH_GPT_total_carpets_l1924_192429

theorem total_carpets 
(house1 : ℕ) 
(house2 : ℕ) 
(house3 : ℕ) 
(house4 : ℕ) 
(h1 : house1 = 12) 
(h2 : house2 = 20) 
(h3 : house3 = 10) 
(h4 : house4 = 2 * house3) : 
  house1 + house2 + house3 + house4 = 62 := 
by 
  -- The proof will be inserted here
  sorry

end NUMINAMATH_GPT_total_carpets_l1924_192429


namespace NUMINAMATH_GPT_difference_in_ages_l1924_192410

variables (J B : ℕ)

-- The conditions: Jack's age is twice Bill's age, and in eight years, Jack will be three times Bill's age then.
axiom condition1 : J = 2 * B
axiom condition2 : J + 8 = 3 * (B + 8)

-- The theorem statement we are proving: The difference in their current ages is 16.
theorem difference_in_ages : J - B = 16 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_ages_l1924_192410


namespace NUMINAMATH_GPT_find_divisor_l1924_192417

theorem find_divisor :
  ∃ D : ℝ, 527652 = (D * 392.57) + 48.25 ∧ D = 1344.25 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1924_192417
