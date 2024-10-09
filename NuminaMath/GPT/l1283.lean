import Mathlib

namespace possible_values_of_angle_F_l1283_128304

-- Define angle F conditions in a triangle DEF
def triangle_angle_F_conditions (D E : ℝ) : Prop :=
  5 * Real.sin D + 2 * Real.cos E = 8 ∧ 3 * Real.sin E + 5 * Real.cos D = 2

-- The main statement: proving the possible values of ∠F
theorem possible_values_of_angle_F (D E : ℝ) (h : triangle_angle_F_conditions D E) : 
  ∃ F : ℝ, F = Real.arcsin (43 / 50) ∨ F = 180 - Real.arcsin (43 / 50) :=
by
  sorry

end possible_values_of_angle_F_l1283_128304


namespace johns_initial_bench_press_weight_l1283_128347

noncomputable def initialBenchPressWeight (currentWeight: ℝ) (injuryPercentage: ℝ) (trainingFactor: ℝ) :=
  (currentWeight / (injuryPercentage / 100 * trainingFactor))

theorem johns_initial_bench_press_weight:
  (initialBenchPressWeight 300 80 3) = 500 :=
by
  sorry

end johns_initial_bench_press_weight_l1283_128347


namespace find_natural_numbers_l1283_128369

noncomputable def valid_n (n : ℕ) : Prop :=
  2 ^ n % 7 = n ^ 2 % 7

theorem find_natural_numbers :
  {n : ℕ | valid_n n} = {n : ℕ | n % 21 = 2 ∨ n % 21 = 4 ∨ n % 21 = 5 ∨ n % 21 = 6 ∨ n % 21 = 10 ∨ n % 21 = 15} :=
sorry

end find_natural_numbers_l1283_128369


namespace algebraic_expression_value_l1283_128315

theorem algebraic_expression_value (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 - 5 * a + 2 = 0) (h3 : b^2 - 5 * b + 2 = 0) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -13 / 2 := by
  sorry

end algebraic_expression_value_l1283_128315


namespace price_reduction_equation_l1283_128341

theorem price_reduction_equation (x : ℝ) :
  63800 * (1 - x)^2 = 3900 :=
sorry

end price_reduction_equation_l1283_128341


namespace initial_average_is_correct_l1283_128306

def initial_average_daily_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) (initial_average : ℕ) :=
  let total_initial_production := initial_average * n
  let total_new_production := total_initial_production + today_production
  let total_days := n + 1
  total_new_production = new_average * total_days

theorem initial_average_is_correct :
  ∀ (A n today_production new_average : ℕ),
    n = 19 →
    today_production = 90 →
    new_average = 52 →
    initial_average_daily_production n today_production new_average A →
    A = 50 := by
    intros A n today_production new_average hn htoday hnew havg
    sorry

end initial_average_is_correct_l1283_128306


namespace chocolates_initial_l1283_128332

variable (x : ℕ)
variable (h1 : 3 * x + 5 + 25 = 5 * x)
variable (h2 : x = 15)

theorem chocolates_initial (x : ℕ) (h1 : 3 * x + 5 + 25 = 5 * x) (h2 : x = 15) : 3 * 15 + 5 = 50 :=
by sorry

end chocolates_initial_l1283_128332


namespace dealership_sedan_sales_l1283_128394

-- Definitions based on conditions:
def sports_cars_ratio : ℕ := 3
def sedans_ratio : ℕ := 5
def anticipated_sports_cars : ℕ := 36

-- Proof problem statement
theorem dealership_sedan_sales :
    (anticipated_sports_cars * sedans_ratio) / sports_cars_ratio = 60 :=
by
  -- Proof goes here
  sorry

end dealership_sedan_sales_l1283_128394


namespace find_value_of_expression_l1283_128361

theorem find_value_of_expression
  (a b : ℝ)
  (h₁ : a = 4 + Real.sqrt 15)
  (h₂ : b = 4 - Real.sqrt 15)
  (h₃ : ∀ x : ℝ, (x^3 - 9 * x^2 + 9 * x = 1) → (x = a ∨ x = b ∨ x = 1))
  : (a / b) + (b / a) = 62 := sorry

end find_value_of_expression_l1283_128361


namespace flour_amount_indeterminable_l1283_128370

variable (flour_required : ℕ)
variable (sugar_required : ℕ := 11)
variable (sugar_added : ℕ := 10)
variable (flour_added : ℕ := 12)
variable (sugar_to_add : ℕ := 1)

theorem flour_amount_indeterminable :
  ¬ ∃ (flour_required : ℕ), flour_additional = flour_required - flour_added :=
by
  sorry

end flour_amount_indeterminable_l1283_128370


namespace find_aroon_pin_l1283_128385

theorem find_aroon_pin (a b : ℕ) (PIN : ℕ) 
  (h0 : 0 ≤ a ∧ a ≤ 9)
  (h1 : 0 ≤ b ∧ b < 1000)
  (h2 : PIN = 1000 * a + b)
  (h3 : 10 * b + a = 3 * PIN - 6) : 
  PIN = 2856 := 
sorry

end find_aroon_pin_l1283_128385


namespace multiple_of_sales_total_l1283_128321

theorem multiple_of_sales_total
  (A : ℝ)
  (M : ℝ)
  (h : M * A = 0.3125 * (11 * A + M * A)) :
  M = 5 :=
by
  sorry

end multiple_of_sales_total_l1283_128321


namespace neg_p_l1283_128357

open Nat -- Opening natural number namespace

-- Definition of the proposition p
def p := ∃ (m : ℕ), ∃ (k : ℕ), k * k = m * m + 1

-- Theorem statement for the negation of proposition p
theorem neg_p : ¬p ↔ ∀ (m : ℕ), ¬ ∃ (k : ℕ), k * k = m * m + 1 :=
by {
  -- Provide the proof here
  sorry
}

end neg_p_l1283_128357


namespace find_coordinates_C_find_range_t_l1283_128352

-- required definitions to handle the given points and vectors
structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

-- Given points
def A : Point := ⟨0, 4⟩
def B : Point := ⟨2, 0⟩

-- Proof for coordinates of point C
theorem find_coordinates_C :
  ∃ C : Point, vector A B = {x := 2 * (C.x - B.x), y := 2 * C.y} ∧ C = ⟨3, -2⟩ :=
by
  sorry

-- Proof for range of t
theorem find_range_t (t : ℝ) :
  let P := Point.mk 3 t
  let PA := vector P A
  let PB := vector P B
  (dot_product PA PB < 0 ∧ -3 * t ≠ -1 * (4 - t)) → 1 < t ∧ t < 3 :=
by
  sorry

end find_coordinates_C_find_range_t_l1283_128352


namespace neg_p_l1283_128313

theorem neg_p (p : ∀ x : ℝ, x^2 ≥ 0) : ∃ x : ℝ, x^2 < 0 := 
sorry

end neg_p_l1283_128313


namespace find_f0_f1_l1283_128314

noncomputable def f : ℤ → ℤ := sorry

theorem find_f0_f1 :
  (∀ x : ℤ, f (x+5) - f x = 10 * x + 25) →
  (∀ x : ℤ, f (x^3 - 1) = (f x - x)^3 + x^3 - 3) →
  f 0 = -1 ∧ f 1 = 0 := by
  intros h1 h2
  sorry

end find_f0_f1_l1283_128314


namespace find_d_l1283_128395

theorem find_d (d : ℝ) (h1 : 0 < d) (h2 : d < 90) (h3 : Real.cos 16 = Real.sin 14 + Real.sin d) : d = 46 :=
by
  sorry

end find_d_l1283_128395


namespace right_triangle_set_l1283_128362

def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem right_triangle_set :
  (is_right_triangle 3 4 2 = false) ∧
  (is_right_triangle 5 12 15 = false) ∧
  (is_right_triangle 8 15 17 = true) ∧
  (is_right_triangle (3^2) (4^2) (5^2) = false) :=
by
  sorry

end right_triangle_set_l1283_128362


namespace sqrt_t6_plus_t4_l1283_128328

open Real

theorem sqrt_t6_plus_t4 (t : ℝ) : sqrt (t^6 + t^4) = t^2 * sqrt (t^2 + 1) :=
by sorry

end sqrt_t6_plus_t4_l1283_128328


namespace range_of_a_l1283_128368

def A (x : ℝ) : Prop := x^2 - 4 * x + 3 ≤ 0
def B (x : ℝ) (a : ℝ) : Prop := x^2 - a * x < x - a

theorem range_of_a (a : ℝ) :
  (∀ x, A x → B x a) ∧ ∃ x, ¬ (A x → B x a) ↔ 1 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l1283_128368


namespace solve_for_y_l1283_128389

theorem solve_for_y (y : ℤ) (h : (y ≠ 2) → ((y^2 - 10*y + 24)/(y-2) + (4*y^2 + 8*y - 48)/(4*y - 8) = 0)) : y = 0 :=
by
  sorry

end solve_for_y_l1283_128389


namespace find_overtime_hours_l1283_128367

theorem find_overtime_hours
  (pay_rate_ordinary : ℝ := 0.60)
  (pay_rate_overtime : ℝ := 0.90)
  (total_pay : ℝ := 32.40)
  (total_hours : ℕ := 50) :
  ∃ y : ℕ, pay_rate_ordinary * (total_hours - y) + pay_rate_overtime * y = total_pay ∧ y = 8 := 
by
  sorry

end find_overtime_hours_l1283_128367


namespace symmetry_xOz_A_l1283_128360

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetry_xOz (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y , z := p.z }

theorem symmetry_xOz_A :
  let A := Point3D.mk 2 (-3) 1
  symmetry_xOz A = Point3D.mk 2 3 1 :=
by
  sorry

end symmetry_xOz_A_l1283_128360


namespace summer_has_150_degrees_l1283_128365

-- Define the condition that Summer has five more degrees than Jolly,
-- and the combined number of degrees they both have is 295.
theorem summer_has_150_degrees (S J : ℕ) (h1 : S = J + 5) (h2 : S + J = 295) : S = 150 :=
by sorry

end summer_has_150_degrees_l1283_128365


namespace cos_expression_value_l1283_128300

theorem cos_expression_value (x : ℝ) (h : Real.sin x = 3 * Real.sin (x - Real.pi / 2)) :
  Real.cos x * Real.cos (x + Real.pi / 2) = 3 / 10 := 
sorry

end cos_expression_value_l1283_128300


namespace inscribed_quadrilateral_exists_l1283_128381

theorem inscribed_quadrilateral_exists (a b c d : ℝ) (h1: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ∃ (p q : ℝ),
    p = Real.sqrt ((a * c + b * d) * (a * d + b * c) / (a * b + c * d)) ∧
    q = Real.sqrt ((a * b + c * d) * (a * d + b * c) / (a * c + b * d)) ∧
    a * c + b * d = p * q :=
by
  sorry

end inscribed_quadrilateral_exists_l1283_128381


namespace sum_of_pairwise_products_does_not_end_in_2019_l1283_128320

theorem sum_of_pairwise_products_does_not_end_in_2019 (n : ℤ) : ¬ (∃ (k : ℤ), 10000 ∣ (3 * n ^ 2 - 2020 + k * 10000)) := by
  sorry

end sum_of_pairwise_products_does_not_end_in_2019_l1283_128320


namespace polynomial_strictly_monotone_l1283_128301

def strictly_monotone (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem polynomial_strictly_monotone
  (P : ℝ → ℝ)
  (H1 : strictly_monotone (P ∘ P))
  (H2 : strictly_monotone (P ∘ P ∘ P)) :
  strictly_monotone P :=
sorry

end polynomial_strictly_monotone_l1283_128301


namespace prob_not_answered_after_three_rings_l1283_128329

def prob_first_ring_answered := 0.1
def prob_second_ring_answered := 0.25
def prob_third_ring_answered := 0.45

theorem prob_not_answered_after_three_rings : 
  1 - prob_first_ring_answered - prob_second_ring_answered - prob_third_ring_answered = 0.2 :=
by
  sorry

end prob_not_answered_after_three_rings_l1283_128329


namespace max_rectangle_area_l1283_128376

noncomputable def curve_parametric_equation (θ : ℝ) :
    ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem max_rectangle_area :
  ∃ (θ : ℝ), (θ ∈ Set.Icc 0 (2 * Real.pi)) ∧
  ∀ (x y : ℝ), (x, y) = curve_parametric_equation θ →
  |(1 + 2 * Real.cos θ) * (1 + 2 * Real.sin θ)| = 3 + 2 * Real.sqrt 2 :=
sorry

end max_rectangle_area_l1283_128376


namespace imaginary_part_div_l1283_128374

open Complex

theorem imaginary_part_div (z1 z2 : ℂ) (h1 : z1 = 1 + I) (h2 : z2 = I) :
  Complex.im (z1 / z2) = -1 := by
  sorry

end imaginary_part_div_l1283_128374


namespace reunion_handshakes_l1283_128371

-- Condition: Number of boys in total
def total_boys : ℕ := 12

-- Condition: Number of left-handed boys
def left_handed_boys : ℕ := 4

-- Condition: Number of right-handed (not exclusively left-handed) boys
def right_handed_boys : ℕ := total_boys - left_handed_boys

-- Function to calculate combinations n choose 2 (number of handshakes in a group)
def combinations (n : ℕ) : ℕ := n * (n - 1) / 2

-- Condition: Number of handshakes among left-handed boys
def handshakes_left (n : ℕ) : ℕ := combinations left_handed_boys

-- Condition: Number of handshakes among right-handed boys
def handshakes_right (n : ℕ) : ℕ := combinations right_handed_boys

-- Problem statement: total number of handshakes
def total_handshakes (total_boys left_handed_boys right_handed_boys : ℕ) : ℕ :=
  handshakes_left left_handed_boys + handshakes_right right_handed_boys

theorem reunion_handshakes : total_handshakes total_boys left_handed_boys right_handed_boys = 34 :=
by sorry

end reunion_handshakes_l1283_128371


namespace instantaneous_velocity_at_2_l1283_128345

def displacement (t : ℝ) : ℝ := 2 * t^3

theorem instantaneous_velocity_at_2 :
  let velocity := deriv displacement
  velocity 2 = 24 :=
by
  sorry

end instantaneous_velocity_at_2_l1283_128345


namespace scientific_notation_of_274000000_l1283_128378

theorem scientific_notation_of_274000000 :
  274000000 = 2.74 * 10^8 := by
  sorry

end scientific_notation_of_274000000_l1283_128378


namespace combination_of_10_choose_3_l1283_128379

theorem combination_of_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end combination_of_10_choose_3_l1283_128379


namespace correct_option_D_l1283_128305

def U : Set ℕ := {1, 2, 4, 6, 8}
def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}
def complement_U_B : Set ℕ := {x ∈ U | x ∉ B}

theorem correct_option_D : A ∩ complement_U_B = {1} := by
  sorry

end correct_option_D_l1283_128305


namespace mariel_dogs_count_l1283_128390

theorem mariel_dogs_count (total_legs : ℤ) (num_dog_walkers : ℤ) (legs_per_walker : ℤ) 
  (other_dogs_count : ℤ) (legs_per_dog : ℤ) (mariel_dogs : ℤ) :
  total_legs = 36 →
  num_dog_walkers = 2 →
  legs_per_walker = 2 →
  other_dogs_count = 3 →
  legs_per_dog = 4 →
  mariel_dogs = (total_legs - (num_dog_walkers * legs_per_walker + other_dogs_count * legs_per_dog)) / legs_per_dog →
  mariel_dogs = 5 :=
by
  intros
  sorry

end mariel_dogs_count_l1283_128390


namespace eq_condition_l1283_128318

theorem eq_condition (a : ℝ) :
  (∃ x : ℝ, a * (4 * |x| + 1) = 4 * |x|) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end eq_condition_l1283_128318


namespace overall_average_marks_l1283_128384

theorem overall_average_marks (n P : ℕ) (P_avg F_avg : ℕ) (H_n : n = 120) (H_P : P = 100) (H_P_avg : P_avg = 39) (H_F_avg : F_avg = 15) :
  (P_avg * P + F_avg * (n - P)) / n = 35 := 
by
  sorry

end overall_average_marks_l1283_128384


namespace equation_not_expression_with_unknowns_l1283_128373

def is_equation (expr : String) : Prop :=
  expr = "equation"

def contains_unknowns (expr : String) : Prop :=
  expr = "contains unknowns"

theorem equation_not_expression_with_unknowns (expr : String) (h1 : is_equation expr) (h2 : contains_unknowns expr) : 
  (is_equation expr) = False := 
sorry

end equation_not_expression_with_unknowns_l1283_128373


namespace willows_in_the_park_l1283_128303

theorem willows_in_the_park (W O : ℕ) 
  (h1 : W + O = 83) 
  (h2 : O = W + 11) : 
  W = 36 := 
by 
  sorry

end willows_in_the_park_l1283_128303


namespace sine_difference_l1283_128396

noncomputable def perpendicular_vectors (θ : ℝ) : Prop :=
  let a := (Real.cos θ, -Real.sqrt 3)
  let b := (1, 1 + Real.sin θ)
  a.1 * b.1 + a.2 * b.2 = 0

theorem sine_difference (θ : ℝ) (h : perpendicular_vectors θ) : Real.sin (Real.pi / 6 - θ) = Real.sqrt 3 / 2 :=
by
  sorry

end sine_difference_l1283_128396


namespace find_number_l1283_128342

theorem find_number (n p q : ℝ) (h1 : n / p = 6) (h2 : n / q = 15) (h3 : p - q = 0.3) : n = 3 :=
by
  sorry

end find_number_l1283_128342


namespace nine_sided_polygon_diagonals_l1283_128350

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l1283_128350


namespace scatter_plot_correlation_l1283_128356

noncomputable def correlation_coefficient (points : List (ℝ × ℝ)) : ℝ := sorry

theorem scatter_plot_correlation {points : List (ℝ × ℝ)} 
  (h : ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) :
  correlation_coefficient points = 1 := 
sorry

end scatter_plot_correlation_l1283_128356


namespace scrabble_champions_l1283_128327

noncomputable def num_champions : Nat := 25
noncomputable def male_percentage : Nat := 40
noncomputable def bearded_percentage : Nat := 40
noncomputable def bearded_bald_percentage : Nat := 60
noncomputable def non_bearded_bald_percentage : Nat := 30

theorem scrabble_champions :
  let male_champions := (male_percentage * num_champions) / 100
  let bearded_champions := (bearded_percentage * male_champions) / 100
  let bearded_bald_champions := (bearded_bald_percentage * bearded_champions) / 100
  let bearded_hair_champions := bearded_champions - bearded_bald_champions
  let non_bearded_champions := male_champions - bearded_champions
  let non_bearded_bald_champions := (non_bearded_bald_percentage * non_bearded_champions) / 100
  let non_bearded_hair_champions := non_bearded_champions - non_bearded_bald_champions
  bearded_bald_champions = 2 ∧ 
  bearded_hair_champions = 2 ∧ 
  non_bearded_bald_champions = 1 ∧ 
  non_bearded_hair_champions = 5 :=
by
  sorry

end scrabble_champions_l1283_128327


namespace weight_of_replaced_oarsman_l1283_128377

noncomputable def average_weight (W : ℝ) : ℝ := W / 20

theorem weight_of_replaced_oarsman (W : ℝ) (W_avg : ℝ) (H1 : average_weight W = W_avg) (H2 : average_weight (W + 40) = W_avg + 2) : W = 40 :=
by sorry

end weight_of_replaced_oarsman_l1283_128377


namespace possible_values_of_sum_l1283_128382

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l1283_128382


namespace equivalent_statements_l1283_128339

theorem equivalent_statements (P Q : Prop) : (¬P → Q) ↔ (¬Q → P) :=
by
  sorry

end equivalent_statements_l1283_128339


namespace sample_size_obtained_l1283_128322

/-- A theorem which states the sample size obtained when a sample is taken from a population. -/
theorem sample_size_obtained 
  (total_students : ℕ)
  (sample_students : ℕ)
  (h1 : total_students = 300)
  (h2 : sample_students = 50) : 
  sample_students = 50 :=
by
  sorry

end sample_size_obtained_l1283_128322


namespace vector_addition_l1283_128307

-- Definitions for the vectors
def a : ℝ × ℝ := (5, 2)
def b : ℝ × ℝ := (1, 6)

-- Proof statement (Note: "theorem" is used here instead of "def" because we are stating something to be proven)
theorem vector_addition : a + b = (6, 8) := by
  sorry

end vector_addition_l1283_128307


namespace sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5_l1283_128383

theorem sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5 : Real.sqrt 3 - Real.sqrt 2 > Real.sqrt 6 - Real.sqrt 5 :=
sorry

end sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5_l1283_128383


namespace correct_operation_l1283_128351

theorem correct_operation (a b : ℝ) : 
  (2 * a^2 + a^2 = 3 * a^2) ∧ 
  (a^3 * a^3 ≠ 2 * a^3) ∧ 
  (a^9 / a^3 ≠ a^3) ∧ 
  (¬(7 * a * b - 5 * a = 2)) :=
by 
  sorry

end correct_operation_l1283_128351


namespace boxes_contain_neither_markers_nor_sharpies_l1283_128324

theorem boxes_contain_neither_markers_nor_sharpies :
  (∀ (total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes : ℕ),
    total_boxes = 15 → markers_boxes = 8 → sharpies_boxes = 5 → both_boxes = 4 →
    neither_boxes = total_boxes - (markers_boxes + sharpies_boxes - both_boxes) →
    neither_boxes = 6) :=
by
  intros total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes
  intros htotal hmarkers hsharpies hboth hcalc
  rw [htotal, hmarkers, hsharpies, hboth] at hcalc
  exact hcalc

end boxes_contain_neither_markers_nor_sharpies_l1283_128324


namespace min_class_size_l1283_128388

theorem min_class_size (x : ℕ) (h : 50 ≤ 5 * x + 2) : 52 ≤ 5 * x + 2 :=
by
  sorry

end min_class_size_l1283_128388


namespace negation_of_p_l1283_128319

def p (x : ℝ) : Prop := x^3 - x^2 + 1 < 0

theorem negation_of_p : (¬ ∀ x : ℝ, p x) ↔ ∃ x : ℝ, ¬ p x := by
  sorry

end negation_of_p_l1283_128319


namespace remainder_8_digit_non_decreasing_integers_mod_1000_l1283_128393

noncomputable def M : ℕ :=
  Nat.choose 17 8

theorem remainder_8_digit_non_decreasing_integers_mod_1000 :
  M % 1000 = 310 :=
by
  sorry

end remainder_8_digit_non_decreasing_integers_mod_1000_l1283_128393


namespace three_digit_largest_fill_four_digit_smallest_fill_l1283_128340

theorem three_digit_largest_fill (n : ℕ) (h1 : n * 1000 + 28 * 4 < 1000) : n ≤ 2 := sorry

theorem four_digit_smallest_fill (n : ℕ) (h2 : n * 1000 + 28 * 4 ≥ 1000) : 3 ≤ n := sorry

end three_digit_largest_fill_four_digit_smallest_fill_l1283_128340


namespace translated_point_B_coords_l1283_128354

-- Define the initial point A
def point_A : ℝ × ℝ := (-2, 2)

-- Define the translation operations
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

-- Define the translation of point A to point B
def point_B :=
  translate_right (translate_down point_A 4) 3

-- The proof statement
theorem translated_point_B_coords : point_B = (1, -2) :=
  by sorry

end translated_point_B_coords_l1283_128354


namespace isosceles_triangle_angle_sum_l1283_128353

theorem isosceles_triangle_angle_sum 
  (A B C : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C]
  (AC AB : ℝ) 
  (angle_ABC : ℝ)
  (isosceles : AC = AB)
  (angle_A : angle_ABC = 70) :
  (∃ angle_B : ℝ, angle_B = 55) :=
by
  sorry

end isosceles_triangle_angle_sum_l1283_128353


namespace base8_base9_equivalence_l1283_128323

def base8_digit (x : ℕ) := 0 ≤ x ∧ x < 8
def base9_digit (y : ℕ) := 0 ≤ y ∧ y < 9

theorem base8_base9_equivalence 
    (X Y : ℕ) 
    (hX : base8_digit X) 
    (hY : base9_digit Y) 
    (h_eq : 8 * X + Y = 9 * Y + X) :
    (8 * 7 + 6 = 62) :=
by
  sorry

end base8_base9_equivalence_l1283_128323


namespace point_Q_representation_l1283_128309

-- Definitions
variables {C D Q : Type} [AddCommGroup C] [AddCommGroup D] [AddCommGroup Q] [Module ℝ C] [Module ℝ D] [Module ℝ Q]
variable (CQ : ℝ)
variable (QD : ℝ)
variable (r s : ℝ)

-- Given condition: ratio CQ:QD = 7:2
axiom CQ_QD_ratio : CQ / QD = 7 / 2

-- Proof goal: the affine combination representation of the point Q
theorem point_Q_representation : CQ / (CQ + QD) = 7 / 9 ∧ QD / (CQ + QD) = 2 / 9 :=
sorry

end point_Q_representation_l1283_128309


namespace evaluate_expression_l1283_128391

theorem evaluate_expression : 
  3 * (-3)^4 + 2 * (-3)^3 + (-3)^2 + 3^2 + 2 * 3^3 + 3 * 3^4 = 504 :=
by
  sorry

end evaluate_expression_l1283_128391


namespace small_planters_needed_l1283_128364

-- This states the conditions for the problem
def Oshea_seeds := 200
def large_planters := 4
def large_planter_capacity := 20
def small_planter_capacity := 4
def remaining_seeds := Oshea_seeds - (large_planters * large_planter_capacity) 

-- The target we aim to prove: the number of small planters required
theorem small_planters_needed :
  remaining_seeds / small_planter_capacity = 30 := by
  sorry

end small_planters_needed_l1283_128364


namespace right_triangle_area_l1283_128366

theorem right_triangle_area (a b c : ℝ) (h₀ : a = 24) (h₁ : c = 30) (h2 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 216 :=
by
  sorry

end right_triangle_area_l1283_128366


namespace find_x_given_inverse_relationship_l1283_128386

variable {x y : ℝ}

theorem find_x_given_inverse_relationship 
  (h₀ : x > 0) 
  (h₁ : y > 0) 
  (initial_condition : 3^2 * 25 = 225)
  (inversion_condition : x^2 * y = 225)
  (query : y = 1200) :
  x = Real.sqrt (3 / 16) :=
by
  sorry

end find_x_given_inverse_relationship_l1283_128386


namespace find_x_from_angles_l1283_128343

theorem find_x_from_angles : ∀ (x : ℝ), (6 * x + 3 * x + 2 * x + x = 360) → x = 30 := by
  sorry

end find_x_from_angles_l1283_128343


namespace remainder_of_product_mod_10_l1283_128372

-- Definitions as conditions given in part a
def n1 := 2468
def n2 := 7531
def n3 := 92045

-- The problem expressed as a proof statement
theorem remainder_of_product_mod_10 :
  ((n1 * n2 * n3) % 10) = 0 :=
  by
    -- Sorry is used to skip the proof
    sorry

end remainder_of_product_mod_10_l1283_128372


namespace decreasing_implies_inequality_l1283_128325

variable (f : ℝ → ℝ)

theorem decreasing_implies_inequality (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) : f 3 < f 2 ∧ f 2 < f 1 :=
  sorry

end decreasing_implies_inequality_l1283_128325


namespace rowing_speed_upstream_l1283_128333

theorem rowing_speed_upstream (V_s V_downstream : ℝ) (V_s_eq : V_s = 28) (V_downstream_eq : V_downstream = 31) : 
  V_s - (V_downstream - V_s) = 25 := 
by
  sorry

end rowing_speed_upstream_l1283_128333


namespace machine_shirt_rate_l1283_128349

theorem machine_shirt_rate (S : ℕ) 
  (worked_yesterday : ℕ) (worked_today : ℕ) (shirts_today : ℕ) 
  (h1 : worked_yesterday = 5)
  (h2 : worked_today = 12)
  (h3 : shirts_today = 72)
  (h4 : worked_today * S = shirts_today) : 
  S = 6 := 
by 
  sorry

end machine_shirt_rate_l1283_128349


namespace find_m_l1283_128334

theorem find_m (x n m : ℝ) (h : (x + n)^2 = x^2 + 4*x + m) : m = 4 :=
sorry

end find_m_l1283_128334


namespace multiple_of_rohan_age_l1283_128302

theorem multiple_of_rohan_age (x : ℝ) (h1 : 25 - 15 = 10) (h2 : 25 + 15 = 40) (h3 : 40 = x * 10) : x = 4 := 
by 
  sorry

end multiple_of_rohan_age_l1283_128302


namespace range_of_k_l1283_128399

theorem range_of_k (k : ℝ) (x y : ℝ) : 
  (y = 2 * x - 5 * k + 7) → 
  (y = - (1 / 2) * x + 2) → 
  (x > 0) → 
  (y > 0) → 
  (1 < k ∧ k < 3) :=
by
  sorry

end range_of_k_l1283_128399


namespace common_tangent_y_intercept_l1283_128316

theorem common_tangent_y_intercept
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (m b : ℝ)
  (h_c1 : c1 = (5, -2))
  (h_c2 : c2 = (20, 6))
  (h_r1 : r1 = 5)
  (h_r2 : r2 = 12)
  (h_tangent : ∃m > 0, ∃b, (∀ x y, y = m * x + b → (x - 5)^2 + (y + 2)^2 > 25 ∧ (x - 20)^2 + (y - 6)^2 > 144)) :
  b = -2100 / 161 :=
by
  sorry

end common_tangent_y_intercept_l1283_128316


namespace solution_set_of_inequality_l1283_128331

theorem solution_set_of_inequality (x m : ℝ) : 
  (x^2 - (2 * m + 1) * x + m^2 + m < 0) ↔ m < x ∧ x < m + 1 := 
by
  sorry

end solution_set_of_inequality_l1283_128331


namespace sequence_problem_l1283_128310

theorem sequence_problem :
  7 * 9 * 11 + (7 + 9 + 11) = 720 :=
by
  sorry

end sequence_problem_l1283_128310


namespace find_C_equation_l1283_128348

def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]
def N : Matrix (Fin 2) (Fin 2) ℝ := ![![0, -1], ![1, 0]]

def C2_equation (x y : ℝ) : Prop := y = (1/8) * x^2

theorem find_C_equation (x y : ℝ) :
  (C2_equation (x) y) → (y^2 = 2 * x) := 
sorry

end find_C_equation_l1283_128348


namespace total_weight_of_rings_l1283_128355

theorem total_weight_of_rings :
  let orange_ring := 0.08333333333333333
  let purple_ring := 0.3333333333333333
  let white_ring := 0.4166666666666667
  orange_ring + purple_ring + white_ring = 0.8333333333333333 :=
by
  let orange_ring := 0.08333333333333333
  let purple_ring := 0.3333333333333333
  let white_ring := 0.4166666666666667
  sorry

end total_weight_of_rings_l1283_128355


namespace right_triangle_leg_length_l1283_128380

theorem right_triangle_leg_length (a c b : ℝ) (h : a = 4) (h₁ : c = 5) (h₂ : a^2 + b^2 = c^2) : b = 3 := 
by
  -- by is used for the proof, which we are skipping using sorry.
  sorry

end right_triangle_leg_length_l1283_128380


namespace condition_relation_l1283_128312

variable (A B C : Prop)

theorem condition_relation (h1 : C → B) (h2 : A → B) : 
  (¬(A → C) ∧ ¬(C → A)) :=
by 
  sorry

end condition_relation_l1283_128312


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l1283_128363

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l1283_128363


namespace range_of_values_l1283_128308

variable {f : ℝ → ℝ}

-- Conditions and given data
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x) = f (-x)

def is_monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f (x) ≤ f (y)

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (Real.log a / Real.log 2) + f (-Real.log a / Real.log 2) ≤ 2 * f (1)

-- The goal
theorem range_of_values (h1 : is_even f) (h2 : is_monotone_on_nonneg f) (a : ℝ) (h3 : condition f a) :
  1/2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_values_l1283_128308


namespace valid_selling_price_l1283_128346

-- Define the initial conditions
def cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def sales_increase_per_dollar_decrease : ℝ := 4
def max_profit : ℝ := 13600
def min_selling_price : ℝ := 150

-- Define x as the price reduction per item
variable (x : ℝ)

-- Define the function relationship of the daily sales volume y with respect to x
def sales_volume (x : ℝ) := 100 + 4 * x

-- Define the selling price based on the price reduction
def selling_price (x : ℝ) := 200 - x

-- Calculate the profit based on the selling price and sales volume
def profit (x : ℝ) := (selling_price x - cost_price) * (sales_volume x)

-- Lean theorem statement to prove the given conditions lead to the valid selling price
theorem valid_selling_price (x : ℝ) 
  (h1 : profit x = 13600)
  (h2 : selling_price x ≥ 150) : 
  selling_price x = 185 :=
sorry

end valid_selling_price_l1283_128346


namespace ratio_of_x_to_y_l1283_128330

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x + 2 * y) / (2 * x - y) = 5 / 4) : x / y = -13 / 2 := 
by 
  sorry

end ratio_of_x_to_y_l1283_128330


namespace value_of_x2_minus_y2_l1283_128344

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the conditions
def condition1 : Prop := (x + y) / 2 = 5
def condition2 : Prop := (x - y) / 2 = 2

-- State the theorem to prove
theorem value_of_x2_minus_y2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 40 :=
by
  sorry

end value_of_x2_minus_y2_l1283_128344


namespace cone_volume_l1283_128311

theorem cone_volume (l h : ℝ) (l_eq : l = 5) (h_eq : h = 4) : 
  (1 / 3) * Real.pi * ((l^2 - h^2).sqrt)^2 * h = 12 * Real.pi := 
by 
  sorry

end cone_volume_l1283_128311


namespace money_combination_l1283_128375

variable (Raquel Tom Nataly Sam : ℝ)

-- Given Conditions 
def condition1 : Prop := Tom = (1 / 4) * Nataly
def condition2 : Prop := Nataly = 3 * Raquel
def condition3 : Prop := Sam = 2 * Nataly
def condition4 : Prop := Nataly = (5 / 3) * Sam
def condition5 : Prop := Raquel = 40

-- Proving this combined total
def combined_total : Prop := Tom + Raquel + Nataly + Sam = 262

theorem money_combination (h1: condition1 Tom Nataly) 
                          (h2: condition2 Nataly Raquel) 
                          (h3: condition3 Sam Nataly) 
                          (h4: condition4 Nataly Sam) 
                          (h5: condition5 Raquel) 
                          : combined_total Tom Raquel Nataly Sam :=
sorry

end money_combination_l1283_128375


namespace compute_expression_l1283_128337

theorem compute_expression :
  -9 * 5 - (-(7 * -2)) + (-(11 * -6)) = 7 :=
by
  sorry

end compute_expression_l1283_128337


namespace find_xy_l1283_128387

theorem find_xy : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^4 = y^2 + 71 ∧ x = 6 ∧ y = 35 :=
by
  sorry

end find_xy_l1283_128387


namespace student_B_more_stable_l1283_128326

-- Definitions as stated in the conditions
def student_A_variance : ℝ := 0.3
def student_B_variance : ℝ := 0.1

-- Theorem stating that student B has more stable performance than student A
theorem student_B_more_stable : student_B_variance < student_A_variance :=
by
  sorry

end student_B_more_stable_l1283_128326


namespace more_campers_afternoon_than_morning_l1283_128392

def campers_morning : ℕ := 52
def campers_afternoon : ℕ := 61

theorem more_campers_afternoon_than_morning : campers_afternoon - campers_morning = 9 :=
by
  -- proof goes here
  sorry

end more_campers_afternoon_than_morning_l1283_128392


namespace n_squared_plus_inverse_squared_plus_four_eq_102_l1283_128358

theorem n_squared_plus_inverse_squared_plus_four_eq_102 (n : ℝ) (h : n + 1 / n = 10) :
    n^2 + 1 / n^2 + 4 = 102 :=
by sorry

end n_squared_plus_inverse_squared_plus_four_eq_102_l1283_128358


namespace total_lives_l1283_128338

noncomputable def C : ℝ := 9.5
noncomputable def D : ℝ := C - 3.25
noncomputable def M : ℝ := D + 7.75
noncomputable def E : ℝ := 2 * C - 5.5
noncomputable def F : ℝ := 2/3 * E

theorem total_lives : C + D + M + E + F = 52.25 :=
by
  sorry

end total_lives_l1283_128338


namespace max_belts_l1283_128336

theorem max_belts (h t b : ℕ) (Hh : h >= 1) (Ht : t >= 1) (Hb : b >= 1) (total_cost : 3 * h + 4 * t + 9 * b = 60) : b <= 5 :=
sorry

end max_belts_l1283_128336


namespace exists_nat_n_l1283_128359

theorem exists_nat_n (l : ℕ) (hl : l > 0) : ∃ n : ℕ, n^n + 47 ≡ 0 [MOD 2^l] := by
  sorry

end exists_nat_n_l1283_128359


namespace distance_between_tangent_and_parallel_line_l1283_128398

noncomputable def distance_between_parallel_lines 
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ) 
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) : ℝ :=
sorry

variable (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
variable (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop)

axiom tangent_line_at_point (M : ℝ × ℝ) (C : Set (ℝ × ℝ)) : (ℝ × ℝ → Prop)

theorem distance_between_tangent_and_parallel_line
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) :
  C = { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 } →
  M = (-2, 4) →
  l = tangent_line_at_point M C →
  l1 = { p | a * p.1 + 3 * p.2 + 2 * a = 0 } →
  distance_between_parallel_lines C center r M l a l1 = 12/5 :=
by
  intros hC hM hl hl1
  sorry

end distance_between_tangent_and_parallel_line_l1283_128398


namespace base7_of_2345_l1283_128335

def decimal_to_base7 (n : ℕ) : ℕ :=
  6 * 7^3 + 5 * 7^2 + 6 * 7^1 + 0 * 7^0

theorem base7_of_2345 : decimal_to_base7 2345 = 6560 := by
  sorry

end base7_of_2345_l1283_128335


namespace battery_lasts_12_hours_more_l1283_128397

-- Define the battery consumption rates
def standby_consumption_rate : ℚ := 1 / 36
def active_consumption_rate : ℚ := 1 / 4

-- Define the usage times
def total_time_hours : ℚ := 12
def active_use_time_hours : ℚ := 1.5
def standby_time_hours : ℚ := total_time_hours - active_use_time_hours

-- Define the total battery used during standby and active use
def standby_battery_used : ℚ := standby_time_hours * standby_consumption_rate
def active_battery_used : ℚ := active_use_time_hours * active_consumption_rate
def total_battery_used : ℚ := standby_battery_used + active_battery_used

-- Define the remaining battery
def remaining_battery : ℚ := 1 - total_battery_used

-- Define how long the remaining battery will last on standby
def remaining_standby_time : ℚ := remaining_battery / standby_consumption_rate

-- Theorem stating the correct answer
theorem battery_lasts_12_hours_more :
  remaining_standby_time = 12 := 
sorry

end battery_lasts_12_hours_more_l1283_128397


namespace find_overlapping_area_l1283_128317

-- Definitions based on conditions
def length_total : ℕ := 16
def length_strip1 : ℕ := 9
def length_strip2 : ℕ := 7
def area_only_strip1 : ℚ := 27
def area_only_strip2 : ℚ := 18

-- Widths are the same for both strips, hence areas are proportional to lengths
def area_ratio := (length_strip1 : ℚ) / (length_strip2 : ℚ)

-- The Lean statement to prove the question == answer
theorem find_overlapping_area : 
  ∃ S : ℚ, (area_only_strip1 + S) / (area_only_strip2 + S) = area_ratio ∧ 
              area_only_strip1 + S = area_only_strip1 + 13.5 := 
by 
  sorry

end find_overlapping_area_l1283_128317
