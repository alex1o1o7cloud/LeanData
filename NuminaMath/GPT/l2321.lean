import Mathlib

namespace problem1_problem2_l2321_232102

noncomputable def setA (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def setB : Set ℝ := {x | x^2 - 5 * x + 4 ≤ 0}

theorem problem1 (a : ℝ) (h : a = 1) : setA a ∪ setB = {x | 0 ≤ x ∧ x ≤ 4} := by
  sorry

theorem problem2 (a : ℝ) : (∀ x, x ∈ setA a → x ∈ setB) ↔ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end problem1_problem2_l2321_232102


namespace maximum_value_l2321_232141

variable {a b c : ℝ}

-- Conditions
variable (h : a^2 + b^2 = c^2)

theorem maximum_value (h : a^2 + b^2 = c^2) : 
  (∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ 
   (∀ x y z : ℝ, x^2 + y^2 = z^2 → (x^2 + y^2 + x*y) / z^2 ≤ 1.5)) := 
sorry

end maximum_value_l2321_232141


namespace remainder_product_div_10_l2321_232164

def unitsDigit (n : ℕ) : ℕ := n % 10

theorem remainder_product_div_10 :
  let a := 1734
  let b := 5389
  let c := 80607
  let p := a * b * c
  unitsDigit p = 2 := by
  sorry

end remainder_product_div_10_l2321_232164


namespace percent_y_of_x_l2321_232198

theorem percent_y_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y / x = 1 / 3 :=
by
  -- proof steps would be provided here
  sorry

end percent_y_of_x_l2321_232198


namespace boxes_needed_to_sell_l2321_232123

theorem boxes_needed_to_sell (total_bars : ℕ) (bars_per_box : ℕ) (target_boxes : ℕ) (h₁ : total_bars = 710) (h₂ : bars_per_box = 5) : target_boxes = 142 :=
by
  sorry

end boxes_needed_to_sell_l2321_232123


namespace mowing_time_approximately_correct_l2321_232132

noncomputable def timeToMowLawn 
  (length width : ℝ) -- dimensions of the lawn in feet
  (swath overlap : ℝ) -- swath width and overlap in inches
  (speed : ℝ) : ℝ :=  -- walking speed in feet per hour
  (length * (width / ((swath - overlap) / 12))) / speed

theorem mowing_time_approximately_correct
  (h_length : ∀ (length : ℝ), length = 100)
  (h_width : ∀ (width : ℝ), width = 120)
  (h_swath : ∀ (swath : ℝ), swath = 30)
  (h_overlap : ∀ (overlap : ℝ), overlap = 6)
  (h_speed : ∀ (speed : ℝ), speed = 4500) :
  abs (timeToMowLawn 100 120 30 6 4500 - 1.33) < 0.01 := -- assert the answer is approximately 1.33 with a tolerance
by
  intros
  have length := h_length 100
  have width := h_width 120
  have swath := h_swath 30
  have overlap := h_overlap 6
  have speed := h_speed 4500
  rw [length, width, swath, overlap, speed]
  simp [timeToMowLawn]
  sorry

end mowing_time_approximately_correct_l2321_232132


namespace find_k_l2321_232161

def vector := ℝ × ℝ  -- Define a vector as a pair of real numbers

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a (k : ℝ) : vector := (k, 3)
def b : vector := (1, 4)
def c : vector := (2, 1)
def linear_combination (k : ℝ) : vector := ((2 * k - 3), -6)

theorem find_k (k : ℝ) (h : dot_product (linear_combination k) c = 0) : k = 3 := by
  sorry

end find_k_l2321_232161


namespace root_in_interval_l2321_232127

noncomputable def f (x : ℝ) := x^2 + 12 * x - 15

theorem root_in_interval :
  (f 1.1 = -0.59) → (f 1.2 = 0.84) →
  ∃ c, 1.1 < c ∧ c < 1.2 ∧ f c = 0 :=
by
  intros h1 h2
  let h3 := h1
  let h4 := h2
  sorry

end root_in_interval_l2321_232127


namespace marbles_leftover_l2321_232109

theorem marbles_leftover (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) : (r + p) % 8 = 4 :=
by
  sorry

end marbles_leftover_l2321_232109


namespace stephen_female_worker_ants_l2321_232188

-- Define the conditions
def stephen_ants : ℕ := 110
def worker_ants (total_ants : ℕ) : ℕ := total_ants / 2
def male_worker_ants (workers : ℕ) : ℕ := (20 / 100) * workers

-- Define the question and correct answer
def female_worker_ants (total_ants : ℕ) : ℕ :=
  let workers := worker_ants total_ants
  workers - male_worker_ants workers

-- The theorem to prove
theorem stephen_female_worker_ants : female_worker_ants stephen_ants = 44 :=
  by sorry -- Skip the proof for now

end stephen_female_worker_ants_l2321_232188


namespace fraction_of_house_painted_l2321_232155

theorem fraction_of_house_painted (total_time : ℝ) (paint_time : ℝ) (house : ℝ) (h1 : total_time = 60) (h2 : paint_time = 15) (h3 : house = 1) : 
  (paint_time / total_time) * house = 1 / 4 :=
by
  sorry

end fraction_of_house_painted_l2321_232155


namespace sum_of_first_11_terms_of_arithmetic_seq_l2321_232130

noncomputable def arithmetic_sequence_SUM (a d : ℚ) : ℚ :=  
  11 / 2 * (2 * a + 10 * d)

theorem sum_of_first_11_terms_of_arithmetic_seq
  (a d : ℚ)
  (h : a + 2 * d + a + 6 * d = 16) :
  arithmetic_sequence_SUM a d = 88 := 
  sorry

end sum_of_first_11_terms_of_arithmetic_seq_l2321_232130


namespace common_point_of_geometric_progression_l2321_232182

theorem common_point_of_geometric_progression (a b c x y : ℝ) (r : ℝ) 
  (h1 : b = a * r) (h2 : c = a * r^2) 
  (h3 : a * x + b * y = c) : 
  x = 1 / 2 ∧ y = -1 / 2 := 
sorry

end common_point_of_geometric_progression_l2321_232182


namespace average_salary_l2321_232196

theorem average_salary (a b c d e : ℕ) (h₁ : a = 8000) (h₂ : b = 5000) (h₃ : c = 15000) (h₄ : d = 7000) (h₅ : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by sorry

end average_salary_l2321_232196


namespace fractions_sum_identity_l2321_232154

theorem fractions_sum_identity (a b c : ℝ) (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / ((b - c) ^ 2) + b / ((c - a) ^ 2) + c / ((a - b) ^ 2) = 0 :=
by
  sorry

end fractions_sum_identity_l2321_232154


namespace function_relation_l2321_232167

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem function_relation:
  f (-Real.pi / 3) > f 1 ∧ f 1 > f (Real.pi / 5) :=
by 
  sorry

end function_relation_l2321_232167


namespace simplify_rationalize_l2321_232101

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end simplify_rationalize_l2321_232101


namespace initial_balloons_blown_up_l2321_232124
-- Import necessary libraries

-- Define the statement
theorem initial_balloons_blown_up (x : ℕ) (hx : x + 13 = 60) : x = 47 :=
by
  sorry

end initial_balloons_blown_up_l2321_232124


namespace smallest_possible_area_square_l2321_232171

theorem smallest_possible_area_square : 
  ∃ (c : ℝ), (∀ (x y : ℝ), ((y = 3 * x - 20) ∨ (y = x^2)) ∧ 
      (10 * (9 + 4 * c) = ((c + 20) / Real.sqrt 10) ^ 2) ∧ 
      (c = 80) ∧ 
      (10 * (9 + 4 * c) = 3290)) :=
by {
  use 80,
  sorry
}

end smallest_possible_area_square_l2321_232171


namespace jillian_apartment_size_l2321_232172

theorem jillian_apartment_size :
  ∃ (s : ℝ), (1.20 * s = 720) ∧ s = 600 := by
sorry

end jillian_apartment_size_l2321_232172


namespace rabbit_excursion_time_l2321_232126

theorem rabbit_excursion_time 
  (line_length : ℝ := 40) 
  (line_speed : ℝ := 3) 
  (rabbit_speed : ℝ := 5) : 
  -- The time calculated for the rabbit to return is 25 seconds
  (line_length / (rabbit_speed - line_speed) + line_length / (rabbit_speed + line_speed)) = 25 :=
by
  -- Placeholder for the proof, to be filled in with a detailed proof later
  sorry

end rabbit_excursion_time_l2321_232126


namespace first_marvelous_monday_after_school_starts_l2321_232156

def is_marvelous_monday (year : ℕ) (month : ℕ) (day : ℕ) (start_day : ℕ) : Prop :=
  let days_in_month := if month = 9 then 30 else if month = 10 then 31 else 0
  let fifth_monday := start_day + 28
  let is_monday := (fifth_monday - 1) % 7 = 0
  month = 10 ∧ day = 30 ∧ is_monday

theorem first_marvelous_monday_after_school_starts :
  ∃ (year month day : ℕ),
    year = 2023 ∧ month = 10 ∧ day = 30 ∧ is_marvelous_monday year month day 4 := sorry

end first_marvelous_monday_after_school_starts_l2321_232156


namespace range_of_a_l2321_232173

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) is an odd and monotonically increasing function, to be defined later.

noncomputable def g (x a : ℝ) : ℝ :=
  f (x^2) + f (a - 2 * |x|)

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0 ∧ g x4 a = 0 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ↔
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l2321_232173


namespace stone_145_is_5_l2321_232105

theorem stone_145_is_5 :
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 15) → (145 % 28) = 5 → n = 5 :=
by
  intros n h h145
  sorry

end stone_145_is_5_l2321_232105


namespace order_of_magnitude_l2321_232176

theorem order_of_magnitude (a b : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : |a| < |b|) :
  -b > a ∧ a > -a ∧ -a > b := by
  sorry

end order_of_magnitude_l2321_232176


namespace solve_congruence_l2321_232135

theorem solve_congruence (x : ℤ) : 
  (10 * x + 3) % 18 = 11 % 18 → x % 9 = 8 % 9 :=
by {
  sorry
}

end solve_congruence_l2321_232135


namespace sin_cos_75_eq_quarter_l2321_232143

theorem sin_cos_75_eq_quarter : (Real.sin (75 * Real.pi / 180)) * (Real.cos (75 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end sin_cos_75_eq_quarter_l2321_232143


namespace students_in_band_l2321_232142

theorem students_in_band (total_students : ℕ) (band_percentage : ℚ) (h_total_students : total_students = 840) (h_band_percentage : band_percentage = 0.2) : ∃ band_students : ℕ, band_students = 168 ∧ band_students = band_percentage * total_students := 
sorry

end students_in_band_l2321_232142


namespace sum_even_odd_functions_l2321_232157

theorem sum_even_odd_functions (f g : ℝ → ℝ) (h₁ : ∀ x, f (-x) = f x) (h₂ : ∀ x, g (-x) = -g x) (h₃ : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  f 1 + g 1 = 1 := 
by 
  sorry

end sum_even_odd_functions_l2321_232157


namespace initial_bananas_per_child_l2321_232192

theorem initial_bananas_per_child 
    (absent : ℕ) (present : ℕ) (total : ℕ) (x : ℕ) (B : ℕ)
    (h1 : absent = 305)
    (h2 : present = 305)
    (h3 : total = 610)
    (h4 : B = present * (x + 2))
    (h5 : B = total * x) : 
    x = 2 :=
by
  sorry

end initial_bananas_per_child_l2321_232192


namespace find_number_l2321_232111

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
sorry

end find_number_l2321_232111


namespace photos_per_week_in_february_l2321_232117

def january_photos : ℕ := 31 * 2

def total_photos (jan_feb_photos : ℕ) : ℕ := jan_feb_photos - january_photos

theorem photos_per_week_in_february (jan_feb_photos : ℕ) (weeks_in_february : ℕ)
  (h1 : jan_feb_photos = 146)
  (h2 : weeks_in_february = 4) :
  total_photos jan_feb_photos / weeks_in_february = 21 := by
  sorry

end photos_per_week_in_february_l2321_232117


namespace smallest_four_digit_multiple_of_18_l2321_232119

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end smallest_four_digit_multiple_of_18_l2321_232119


namespace elvins_first_month_bill_l2321_232165

-- Define the variables involved
variables (F C : ℝ)

-- State the given conditions
def condition1 : Prop := F + C = 48
def condition2 : Prop := F + 2 * C = 90

-- State the theorem we need to prove
theorem elvins_first_month_bill (F C : ℝ) (h1 : F + C = 48) (h2 : F + 2 * C = 90) : F + C = 48 :=
by sorry

end elvins_first_month_bill_l2321_232165


namespace min_value_3_div_a_add_2_div_b_l2321_232129

/-- Given positive real numbers a and b, and the condition that the lines
(a + 1)x + 2y - 1 = 0 and 3x + (b - 2)y + 2 = 0 are perpendicular,
prove that the minimum value of 3/a + 2/b is 25, given the condition 3a + 2b = 1. -/
theorem min_value_3_div_a_add_2_div_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h : 3 * a + 2 * b = 1) : 3 / a + 2 / b ≥ 25 :=
sorry

end min_value_3_div_a_add_2_div_b_l2321_232129


namespace petya_time_l2321_232128

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l2321_232128


namespace cost_per_day_is_18_l2321_232149

def cost_per_day_first_week (x : ℕ) : Prop :=
  let cost_per_day_rest_week := 12
  let total_days := 23
  let total_cost := 318
  let first_week_days := 7
  let remaining_days := total_days - first_week_days
  (first_week_days * x) + (remaining_days * cost_per_day_rest_week) = total_cost

theorem cost_per_day_is_18 : cost_per_day_first_week 18 :=
  sorry

end cost_per_day_is_18_l2321_232149


namespace trip_time_l2321_232190

theorem trip_time (T : ℝ) (x : ℝ) : 
  (150 / 4 = 50 / 30 + (x - 50) / 4 + (150 - x) / 30) → (T = 37.5) :=
by
  sorry

end trip_time_l2321_232190


namespace divisor_of_109_l2321_232179

theorem divisor_of_109 (d : ℕ) (h : 109 = 9 * d + 1) : d = 12 :=
sorry

end divisor_of_109_l2321_232179


namespace find_height_on_BC_l2321_232185

noncomputable def height_on_BC (a b : ℝ) (A B C : ℝ) : ℝ := b * (Real.sin C)

theorem find_height_on_BC (A B C a b h : ℝ)
  (h_a: a = Real.sqrt 3)
  (h_b: b = Real.sqrt 2)
  (h_cos: 1 + 2 * Real.cos (B + C) = 0)
  (h_A: A = Real.pi / 3)
  (h_B: B = Real.pi / 4)
  (h_C: C = 5 * Real.pi / 12)
  (h_h: h = height_on_BC a b A B C) :
  h = (Real.sqrt 3 + 1) / 2 :=
sorry

end find_height_on_BC_l2321_232185


namespace number_of_girls_l2321_232106

-- Define the number of girls and boys
variables (G B : ℕ)

-- Define the conditions
def condition1 : Prop := B = 2 * G - 16
def condition2 : Prop := G + B = 68

-- The theorem we want to prove
theorem number_of_girls (h1 : condition1 G B) (h2 : condition2 G B) : G = 28 :=
by
  sorry

end number_of_girls_l2321_232106


namespace eunji_class_total_students_l2321_232139

variable (A B : Finset ℕ) (universe_students : Finset ℕ)

axiom students_play_instrument_a : A.card = 24
axiom students_play_instrument_b : B.card = 17
axiom students_play_both_instruments : (A ∩ B).card = 8
axiom no_students_without_instruments : A ∪ B = universe_students

theorem eunji_class_total_students : universe_students.card = 33 := by
  sorry

end eunji_class_total_students_l2321_232139


namespace total_earnings_correct_l2321_232186

-- Definitions for the conditions
def price_per_bracelet := 5
def price_for_two_bracelets := 8
def initial_bracelets := 30
def earnings_from_selling_at_5_each := 60

-- Variables to store intermediate calculations
def bracelets_sold_at_5_each := earnings_from_selling_at_5_each / price_per_bracelet
def remaining_bracelets := initial_bracelets - bracelets_sold_at_5_each
def pairs_sold_at_8_each := remaining_bracelets / 2
def earnings_from_pairs := pairs_sold_at_8_each * price_for_two_bracelets
def total_earnings := earnings_from_selling_at_5_each + earnings_from_pairs

-- The theorem stating that Zayne made $132 in total
theorem total_earnings_correct :
  total_earnings = 132 :=
sorry

end total_earnings_correct_l2321_232186


namespace Megan_acorns_now_l2321_232168

def initial_acorns := 16
def given_away_acorns := 7
def remaining_acorns := initial_acorns - given_away_acorns

theorem Megan_acorns_now : remaining_acorns = 9 := by
  sorry

end Megan_acorns_now_l2321_232168


namespace ellipse_hyperbola_tangent_l2321_232100

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∃ (x y : ℝ), x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y - 2)^2 = 4) →
  m = 45 / 31 :=
by sorry

end ellipse_hyperbola_tangent_l2321_232100


namespace age_difference_l2321_232137

def JobAge := 5
def StephanieAge := 4 * JobAge
def FreddyAge := 18

theorem age_difference : StephanieAge - FreddyAge = 2 := by
  sorry

end age_difference_l2321_232137


namespace total_exercise_time_l2321_232191

-- Definitions based on given conditions
def javier_daily : ℕ := 50
def javier_days : ℕ := 7
def sanda_daily : ℕ := 90
def sanda_days : ℕ := 3

-- Proof problem to verify the total exercise time for both Javier and Sanda
theorem total_exercise_time : javier_daily * javier_days + sanda_daily * sanda_days = 620 := by
  sorry

end total_exercise_time_l2321_232191


namespace system_of_equations_n_eq_1_l2321_232175

theorem system_of_equations_n_eq_1 {x y n : ℝ} 
  (h₁ : 5 * x - 4 * y = n) 
  (h₂ : 3 * x + 5 * y = 8)
  (h₃ : x = y) : 
  n = 1 := 
by
  sorry

end system_of_equations_n_eq_1_l2321_232175


namespace unique_A3_zero_l2321_232112

variable {F : Type*} [Field F]

theorem unique_A3_zero (A : Matrix (Fin 2) (Fin 2) F) 
  (h1 : A ^ 4 = 0) 
  (h2 : Matrix.trace A = 0) : 
  A ^ 3 = 0 :=
sorry

end unique_A3_zero_l2321_232112


namespace least_distance_fly_crawled_l2321_232174

noncomputable def leastDistance (baseRadius height startDist endDist : ℝ) : ℝ :=
  let C := 2 * Real.pi * baseRadius
  let slantHeight := Real.sqrt (baseRadius ^ 2 + height ^ 2)
  let theta := C / slantHeight
  let x1 := startDist * Real.cos 0
  let y1 := startDist * Real.sin 0
  let x2 := endDist * Real.cos (theta / 2)
  let y2 := endDist * Real.sin (theta / 2)
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem least_distance_fly_crawled (baseRadius height startDist endDist : ℝ) (h1 : baseRadius = 500) (h2 : height = 150 * Real.sqrt 7) (h3 : startDist = 150) (h4 : endDist = 300 * Real.sqrt 2) :
  leastDistance baseRadius height startDist endDist = 150 * Real.sqrt 13 := by
  sorry

end least_distance_fly_crawled_l2321_232174


namespace jacob_dimes_l2321_232163

-- Definitions of the conditions
def mrs_hilt_total_cents : ℕ := 2 * 1 + 2 * 10 + 2 * 5
def jacob_base_cents : ℕ := 4 * 1 + 1 * 5
def difference : ℕ := 13

-- The proof problem: prove Jacob has 1 dime.
theorem jacob_dimes (d : ℕ) (h : mrs_hilt_total_cents - (jacob_base_cents + 10 * d) = difference) : d = 1 := by
  sorry

end jacob_dimes_l2321_232163


namespace obtuse_angle_condition_l2321_232152

def dot_product (a b : (ℝ × ℝ)) : ℝ := a.1 * b.1 + a.2 * b.2

def is_obtuse_angle (a b : (ℝ × ℝ)) : Prop := dot_product a b < 0

def is_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem obtuse_angle_condition :
  (∀ (x : ℝ), x > 0 → is_obtuse_angle (-1, 0) (x, 1 - x) ∧ ¬is_parallel (-1, 0) (x, 1 - x)) ∧ 
  (∀ (x : ℝ), is_obtuse_angle (-1, 0) (x, 1 - x) → x > 0) :=
sorry

end obtuse_angle_condition_l2321_232152


namespace percentage_increase_is_20_l2321_232158

noncomputable def total_stocks : ℕ := 1980
noncomputable def stocks_higher : ℕ := 1080
noncomputable def stocks_lower : ℕ := total_stocks - stocks_higher

/--
Given that the total number of stocks is 1,980, and 1,080 stocks closed at a higher price today than yesterday.
Furthermore, the number of stocks that closed higher today is greater than the number that closed lower.

Prove that the percentage increase in the number of stocks that closed at a higher price today compared to the number that closed at a lower price is 20%.
-/
theorem percentage_increase_is_20 :
  (stocks_higher - stocks_lower) / stocks_lower * 100 = 20 := by
  sorry

end percentage_increase_is_20_l2321_232158


namespace orthogonal_planes_k_value_l2321_232181

theorem orthogonal_planes_k_value
  (k : ℝ)
  (h : 3 * (-1) + 1 * 1 + (-2) * k = 0) : 
  k = -1 :=
sorry

end orthogonal_planes_k_value_l2321_232181


namespace find_a6_l2321_232170

theorem find_a6 (a : ℕ → ℚ) (h₁ : ∀ n, a (n + 1) = 2 * a n - 1) (h₂ : a 8 = 16) : a 6 = 19 / 4 :=
sorry

end find_a6_l2321_232170


namespace proof_of_calculation_l2321_232199

theorem proof_of_calculation : (7^2 - 5^2)^4 = 331776 := by
  sorry

end proof_of_calculation_l2321_232199


namespace dodecahedron_edge_probability_l2321_232116

def numVertices := 20
def pairsChosen := Nat.choose 20 2  -- Calculates combination (20 choose 2)
def edgesPerVertex := 3
def numEdges := (numVertices * edgesPerVertex) / 2
def probability : ℚ := numEdges / pairsChosen

theorem dodecahedron_edge_probability :
  probability = 3 / 19 :=
by
  -- The proof is skipped as per the instructions
  sorry

end dodecahedron_edge_probability_l2321_232116


namespace linear_function_quadrants_l2321_232108

theorem linear_function_quadrants (m : ℝ) (h1 : m - 2 < 0) (h2 : m + 1 > 0) : -1 < m ∧ m < 2 := 
by 
  sorry

end linear_function_quadrants_l2321_232108


namespace problem_statement_l2321_232150

theorem problem_statement (n : ℕ) (a b c : ℕ → ℤ)
  (h1 : n > 0)
  (h2 : ∀ i j, i ≠ j → ¬ (a i - a j) % n = 0 ∧
                           ¬ ((b i + c i) - (b j + c j)) % n = 0 ∧
                           ¬ (b i - b j) % n = 0 ∧
                           ¬ ((c i + a i) - (c j + a i)) % n = 0 ∧
                           ¬ (c i - c j) % n = 0 ∧
                           ¬ ((a i + b i) - (a j + b i)) % n = 0 ∧
                           ¬ ((a i + b i + c i) - (a j + b i + c j)) % n = 0) :
  (Odd n) ∧ (¬ ∃ k, n = 3 * k) :=
by sorry

end problem_statement_l2321_232150


namespace sin_double_angle_l2321_232104

-- Given Conditions
variable {α : ℝ}
variable (h1 : 0 < α ∧ α < π / 2) -- α is in the first quadrant
variable (h2 : Real.sin α = 3 / 5) -- sin(α) = 3/5

-- Theorem statement
theorem sin_double_angle (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 := 
sorry

end sin_double_angle_l2321_232104


namespace smallest_n_for_multiple_of_5_l2321_232103

theorem smallest_n_for_multiple_of_5 (x y : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 5]) (h2 : y - 2 ≡ 0 [ZMOD 5]) :
  ∃ n : ℕ, n > 0 ∧ x^2 + x * y + y^2 + n ≡ 0 [ZMOD 5] ∧ n = 1 := 
sorry

end smallest_n_for_multiple_of_5_l2321_232103


namespace sum_of_prime_factors_l2321_232193

theorem sum_of_prime_factors (n : ℕ) (h : n = 257040) : 
  (2 + 5 + 3 + 107 = 117) :=
by sorry

end sum_of_prime_factors_l2321_232193


namespace leo_current_weight_l2321_232110

variables (L K J : ℝ)

def condition1 := L + 12 = 1.7 * K
def condition2 := L + K + J = 270
def condition3 := J = K + 30

theorem leo_current_weight (h1 : condition1 L K)
                           (h2 : condition2 L K J)
                           (h3 : condition3 K J) : L = 103.6 :=
sorry

end leo_current_weight_l2321_232110


namespace common_root_l2321_232145

def f (x : ℝ) : ℝ := x^4 - x^3 - 22 * x^2 + 16 * x + 96
def g (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 10

theorem common_root :
  f (-2) = 0 ∧ g (-2) = 0 := by
  sorry

end common_root_l2321_232145


namespace prob_Z_l2321_232134

theorem prob_Z (P_X P_Y P_W P_Z : ℚ) (hX : P_X = 1/4) (hY : P_Y = 1/3) (hW : P_W = 1/6) 
(hSum : P_X + P_Y + P_Z + P_W = 1) : P_Z = 1/4 := 
by
  -- The proof will be filled in later
  sorry

end prob_Z_l2321_232134


namespace problem1_problem2_l2321_232184

theorem problem1 : 
  (5 / 7 : ℚ) * (-14 / 3) / (5 / 3) = -2 := 
by 
  sorry

theorem problem2 : 
  (-15 / 7 : ℚ) / (-6 / 5) * (-7 / 5) = -5 / 2 := 
by 
  sorry

end problem1_problem2_l2321_232184


namespace power_log_simplification_l2321_232107

theorem power_log_simplification (x : ℝ) (h : x > 0) : (16^(Real.log x / Real.log 2))^(1/4) = x :=
by sorry

end power_log_simplification_l2321_232107


namespace balcony_more_than_orchestra_l2321_232194

theorem balcony_more_than_orchestra (O B : ℕ) 
  (h1 : O + B = 355) 
  (h2 : 12 * O + 8 * B = 3320) : 
  B - O = 115 :=
by 
  -- Sorry, this will skip the proof.
  sorry

end balcony_more_than_orchestra_l2321_232194


namespace baking_time_one_batch_l2321_232197

theorem baking_time_one_batch (x : ℕ) (time_icing_per_batch : ℕ) (num_batches : ℕ) (total_time : ℕ)
  (h1 : num_batches = 4)
  (h2 : time_icing_per_batch = 30)
  (h3 : total_time = 200)
  (h4 : total_time = num_batches * x + num_batches * time_icing_per_batch) :
  x = 20 :=
by
  rw [h1, h2, h3] at h4
  sorry

end baking_time_one_batch_l2321_232197


namespace line_parallel_condition_l2321_232118

theorem line_parallel_condition (a : ℝ) : (a = 2) ↔ (∀ x y : ℝ, (ax + 2 * y = 0 → x + y ≠ 1)) :=
by
  sorry

end line_parallel_condition_l2321_232118


namespace price_per_foot_of_fence_l2321_232113

theorem price_per_foot_of_fence (area : ℝ) (total_cost : ℝ) (side_length : ℝ) (perimeter : ℝ) (price_per_foot : ℝ) 
  (h1 : area = 289) (h2 : total_cost = 3672) (h3 : side_length = Real.sqrt area) (h4 : perimeter = 4 * side_length) (h5 : price_per_foot = total_cost / perimeter) :
  price_per_foot = 54 := by
  sorry

end price_per_foot_of_fence_l2321_232113


namespace analysis_method_correct_answer_l2321_232138

axiom analysis_def (conclusion: Prop): 
  ∃ sufficient_conditions: (Prop → Prop), 
    (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)

theorem analysis_method_correct_answer :
  ∀ (conclusion : Prop) , ∃ sufficient_conditions: (Prop → Prop), 
  (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)
:= by 
  intros 
  sorry

end analysis_method_correct_answer_l2321_232138


namespace exactly_two_succeed_probability_l2321_232148

-- Define the probabilities of events A, B, and C decrypting the code
def P_A_decrypts : ℚ := 1/5
def P_B_decrypts : ℚ := 1/4
def P_C_decrypts : ℚ := 1/3

-- Define the probabilities of events A, B, and C not decrypting the code
def P_A_not_decrypts : ℚ := 1 - P_A_decrypts
def P_B_not_decrypts : ℚ := 1 - P_B_decrypts
def P_C_not_decrypts : ℚ := 1 - P_C_decrypts

-- Define the probability that exactly two out of A, B, and C decrypt the code
def P_exactly_two_succeed : ℚ :=
  (P_A_decrypts * P_B_decrypts * P_C_not_decrypts) +
  (P_A_decrypts * P_B_not_decrypts * P_C_decrypts) +
  (P_A_not_decrypts * P_B_decrypts * P_C_decrypts)

-- Prove that this probability is equal to 3/20
theorem exactly_two_succeed_probability : P_exactly_two_succeed = 3 / 20 := by
  sorry

end exactly_two_succeed_probability_l2321_232148


namespace gcd_polynomial_l2321_232159

theorem gcd_polynomial {b : ℕ} (h : 570 ∣ b) : Nat.gcd (4*b^3 + 2*b^2 + 5*b + 95) b = 95 := 
sorry

end gcd_polynomial_l2321_232159


namespace arcsin_half_eq_pi_six_arccos_sqrt_three_over_two_eq_pi_six_l2321_232151

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  sorry

theorem arccos_sqrt_three_over_two_eq_pi_six : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arcsin_half_eq_pi_six_arccos_sqrt_three_over_two_eq_pi_six_l2321_232151


namespace cyclist_time_no_wind_l2321_232146

theorem cyclist_time_no_wind (v w : ℝ) 
    (h1 : v + w = 1 / 3) 
    (h2 : v - w = 1 / 4) : 
    1 / v = 24 / 7 := 
by
  sorry

end cyclist_time_no_wind_l2321_232146


namespace probability_comparison_l2321_232195

variables (M N : ℕ) (m n : ℝ)
variable (h₁ : m > 10^6)
variable (h₂ : n ≤ 10^6)

theorem probability_comparison (h₃: 0 < M) (h₄: 0 < N):
  (m * M) / (m * M + n * N) > (M / (M + N)) :=
by
  have h₅: n / m < 1 := sorry
  have h₆: M > 0 := by linarith
  have h₇: 1 + (n / m) * (N / M) < 2 := sorry
  sorry

end probability_comparison_l2321_232195


namespace find_other_digits_l2321_232169

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem find_other_digits (n : ℕ) (h : ℕ) :
  tens_digit n = h →
  h = 1 →
  is_divisible_by_9 n →
  ∃ m : ℕ, m < 9 ∧ n = 10 * ((n / 10) / 10) * 10 + h * 10 + m ∧ (∃ k : ℕ, k * 9 = h + m + (n / 100)) :=
sorry

end find_other_digits_l2321_232169


namespace problem_to_prove_l2321_232120

theorem problem_to_prove
  (a b c : ℝ)
  (h1 : a + b + c = -3)
  (h2 : a * b + b * c + c * a = -10)
  (h3 : a * b * c = -5) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = 70 :=
by
  sorry

end problem_to_prove_l2321_232120


namespace find_m_l2321_232122

theorem find_m (f : ℝ → ℝ) (m : ℝ) 
  (h_even : ∀ x, f (-x) = f x) 
  (h_fx : ∀ x, 0 < x → f x = 4^(m - x)) 
  (h_f_neg2 : f (-2) = 1/8) : 
  m = 1/2 := 
by 
  sorry

end find_m_l2321_232122


namespace unique_real_solution_l2321_232133

theorem unique_real_solution (a : ℝ) : 
  (∀ x : ℝ, (x^3 - a * x^2 - (a + 1) * x + (a^2 - 2) = 0)) ↔ (a < 7 / 4) := 
sorry

end unique_real_solution_l2321_232133


namespace smallest_b_value_l2321_232177

theorem smallest_b_value (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a - b = 7) 
    (h₄ : (Nat.gcd ((a^3 + b^3) / (a + b)) (a^2 * b)) = 12) : b = 6 :=
by
    -- proof goes here
    sorry

end smallest_b_value_l2321_232177


namespace minimum_value_l2321_232153

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y = 1) :
  ∀ (z : ℝ), z = (1/x + 1/y) → z ≥ 3 + 2*Real.sqrt 2 :=
by
  sorry

end minimum_value_l2321_232153


namespace value_of_certain_number_l2321_232189

theorem value_of_certain_number (a b : ℕ) (h : 1 / 7 * 8 = 5) (h2 : 1 / 5 * b = 35) : b = 175 :=
by
  -- by assuming the conditions hold, we need to prove b = 175
  sorry

end value_of_certain_number_l2321_232189


namespace cos_value_given_sin_l2321_232160

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (π / 3 - α) = 3 / 5 :=
sorry

end cos_value_given_sin_l2321_232160


namespace num_terms_in_expansion_eq_3_pow_20_l2321_232166

-- Define the expression 
def expr (x y : ℝ) := (1 + x + y) ^ 20

-- Statement of the problem
theorem num_terms_in_expansion_eq_3_pow_20 (x y : ℝ) : (3 : ℝ)^20 = (1 + x + y) ^ 20 :=
by sorry

end num_terms_in_expansion_eq_3_pow_20_l2321_232166


namespace sum_of_a_and_b_l2321_232125

variables {a b m : ℝ}

theorem sum_of_a_and_b (h1 : a^2 + a * b = 16 + m) (h2 : b^2 + a * b = 9 - m) : a + b = 5 ∨ a + b = -5 :=
by sorry

end sum_of_a_and_b_l2321_232125


namespace original_equation_solution_l2321_232121

noncomputable def original_equation : Prop :=
  ∃ Y P A K P O C : ℕ,
  (Y = 5) ∧ (P = 2) ∧ (A = 0) ∧ (K = 2) ∧ (P = 4) ∧ (O = 0) ∧ (C = 0) ∧
  (Y.factorial * P.factorial * A.factorial = K * 10000 + P * 1000 + O * 100 + C * 10 + C)

theorem original_equation_solution : original_equation :=
  sorry

end original_equation_solution_l2321_232121


namespace first_problem_l2321_232147

-- Definitions for the first problem
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable (h_pos : ∀ n, a n > 0)
variable (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1))

-- Theorem statement for the first problem
theorem first_problem (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1)) :
  ∃ d, ∀ n, a (n + 1) - a n = d := sorry

end first_problem_l2321_232147


namespace cheryl_material_leftover_l2321_232115

theorem cheryl_material_leftover :
  let material1 := (5 / 9 : ℚ)
  let material2 := (1 / 3 : ℚ)
  let total_bought := material1 + material2
  let used := (0.5555555555555556 : ℝ)
  let total_bought_decimal := (8 / 9 : ℝ)
  let leftover := total_bought_decimal - used
  leftover = 0.3333333333333332 := by
sorry

end cheryl_material_leftover_l2321_232115


namespace find_n_for_divisibility_by_33_l2321_232136

theorem find_n_for_divisibility_by_33 (n : ℕ) (hn_range : n < 10) (div11 : (12 - n) % 11 = 0) (div3 : (20 + n) % 3 = 0) : n = 1 :=
by {
  -- Proof steps go here
  sorry
}

end find_n_for_divisibility_by_33_l2321_232136


namespace deepak_present_age_l2321_232144

theorem deepak_present_age (x : ℕ) (Rahul_age Deepak_age : ℕ) 
  (h1 : Rahul_age = 4 * x) (h2 : Deepak_age = 3 * x) 
  (h3 : Rahul_age + 4 = 32) : Deepak_age = 21 := by
  sorry

end deepak_present_age_l2321_232144


namespace simplified_value_l2321_232140

-- Define the operation ∗
def operation (m n p q : ℚ) : ℚ :=
  m * p * (n / q)

-- Prove that the simplified value of 5/4 ∗ 6/2 is 60
theorem simplified_value : operation 5 4 6 2 = 60 :=
by
  sorry

end simplified_value_l2321_232140


namespace min_chocolates_for_most_l2321_232183

theorem min_chocolates_for_most (a b c d : ℕ) (h : a < b ∧ b < c ∧ c < d)
  (h_sum : a + b + c + d = 50) : d ≥ 14 := sorry

end min_chocolates_for_most_l2321_232183


namespace exponent_calculation_l2321_232180

theorem exponent_calculation :
  ((19 ^ 11) / (19 ^ 8) * (19 ^ 3) = 47015881) :=
by
  sorry

end exponent_calculation_l2321_232180


namespace cos_phi_expression_l2321_232114

theorem cos_phi_expression (a b c : ℝ) (φ R : ℝ)
  (habc : a > 0 ∧ b > 0 ∧ c > 0)
  (angles : 2 * φ + 3 * φ + 4 * φ = π)
  (law_of_sines : a / Real.sin (2 * φ) = 2 * R ∧ b / Real.sin (3 * φ) = 2 * R ∧ c / Real.sin (4 * φ) = 2 * R) :
  Real.cos φ = (a + c) / (2 * b) := 
by 
  sorry

end cos_phi_expression_l2321_232114


namespace find_P_l2321_232178

theorem find_P (P Q R S : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S) (h4 : Q ≠ R) (h5 : Q ≠ S) (h6 : R ≠ S)
  (h7 : P > 0) (h8 : Q > 0) (h9 : R > 0) (h10 : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDiff : P - Q = R + S) : P = 12 :=
by
  sorry

end find_P_l2321_232178


namespace volume_of_cone_l2321_232131

theorem volume_of_cone (l : ℝ) (A : ℝ) (r : ℝ) (h : ℝ) : 
  l = 10 → A = 60 * Real.pi → (r = 6) → (h = Real.sqrt (10^2 - 6^2)) → 
  (1 / 3 * Real.pi * r^2 * h) = 96 * Real.pi :=
by
  intros
  -- here the proof would be written
  sorry

end volume_of_cone_l2321_232131


namespace pyarelal_loss_l2321_232187

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (h1 : total_loss = 670) (h2 : 1 / 9 * P + P = 10 / 9 * P):
  (9 / (1 + 9)) * total_loss = 603 :=
by
  sorry

end pyarelal_loss_l2321_232187


namespace total_value_of_coins_l2321_232162

variables {p n : ℕ}

-- Ryan has 17 coins consisting of pennies and nickels
axiom coins_eq : p + n = 17

-- The number of pennies is equal to the number of nickels
axiom pennies_eq_nickels : p = n

-- Prove that the total value of Ryan's coins is 49 cents
theorem total_value_of_coins : (p * 1 + n * 5) = 49 :=
by sorry

end total_value_of_coins_l2321_232162
