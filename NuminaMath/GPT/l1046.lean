import Mathlib

namespace NUMINAMATH_GPT_triangle_inequality_half_perimeter_l1046_104653

theorem triangle_inequality_half_perimeter 
  (a b c : ℝ)
  (h_a : a < b + c)
  (h_b : b < a + c)
  (h_c : c < a + b) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := 
sorry

end NUMINAMATH_GPT_triangle_inequality_half_perimeter_l1046_104653


namespace NUMINAMATH_GPT_break_25_ruble_bill_l1046_104628

theorem break_25_ruble_bill (x y z : ℕ) :
  (x + y + z = 11 ∧ 1 * x + 3 * y + 5 * z = 25) ↔ 
    (x = 4 ∧ y = 7 ∧ z = 0) ∨ 
    (x = 5 ∧ y = 5 ∧ z = 1) ∨ 
    (x = 6 ∧ y = 3 ∧ z = 2) ∨ 
    (x = 7 ∧ y = 1 ∧ z = 3) :=
sorry

end NUMINAMATH_GPT_break_25_ruble_bill_l1046_104628


namespace NUMINAMATH_GPT_calc_expression_l1046_104679

variable {x : ℝ}

theorem calc_expression :
    (2 + 3 * x) * (-2 + 3 * x) = 9 * x ^ 2 - 4 := sorry

end NUMINAMATH_GPT_calc_expression_l1046_104679


namespace NUMINAMATH_GPT_data_a_value_l1046_104618

theorem data_a_value (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a + b + c = 96) : a = 12 :=
by
  sorry

end NUMINAMATH_GPT_data_a_value_l1046_104618


namespace NUMINAMATH_GPT_chord_length_l1046_104616

theorem chord_length (r d : ℝ) (h_r : r = 5) (h_d : d = 4) : 
  ∃ EF : ℝ, EF = 6 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l1046_104616


namespace NUMINAMATH_GPT_number_of_beetles_in_sixth_jar_l1046_104670

theorem number_of_beetles_in_sixth_jar :
  ∃ (x : ℕ), 
      (x + (x+1) + (x+2) + (x+3) + (x+4) + (x+5) + (x+6) + (x+7) + (x+8) + (x+9) = 150) ∧
      (2 * x ≥ x + 9) ∧
      (x + 5 = 16) :=
by {
  -- This is just the statement, the proof steps are ommited.
  -- You can fill in the proof here using Lean tactics as necessary.
  sorry
}

end NUMINAMATH_GPT_number_of_beetles_in_sixth_jar_l1046_104670


namespace NUMINAMATH_GPT_find_a_l1046_104639

noncomputable def curve (x a : ℝ) : ℝ := 1/x + (Real.log x)/a
noncomputable def curve_derivative (x a : ℝ) : ℝ := 
  (-1/(x^2)) + (1/(a * x))

theorem find_a (a : ℝ) : 
  (curve_derivative 1 a = 3/2) ∧ ((∃ l : ℝ, curve 1 a = l) → ∃ m : ℝ, m * (-2/3) = -1)  → a = 2/5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1046_104639


namespace NUMINAMATH_GPT_work_completion_l1046_104673

theorem work_completion (x y : ℕ) : 
  (1 / (x + y) = 1 / 12) ∧ (1 / y = 1 / 24) → x = 24 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_l1046_104673


namespace NUMINAMATH_GPT_project_B_days_l1046_104682

theorem project_B_days (B : ℕ) : 
  (1 / 20 + 1 / B) * 10 + (1 / B) * 5 = 1 -> B = 30 :=
by
  sorry

end NUMINAMATH_GPT_project_B_days_l1046_104682


namespace NUMINAMATH_GPT_algebra_inequality_l1046_104690

theorem algebra_inequality (a b c : ℝ) 
  (H : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_GPT_algebra_inequality_l1046_104690


namespace NUMINAMATH_GPT_roots_ratio_sum_l1046_104698

theorem roots_ratio_sum (α β : ℝ) (hαβ : α > β) (h1 : 3*α^2 + α - 1 = 0) (h2 : 3*β^2 + β - 1 = 0) :
  α / β + β / α = -7 / 3 :=
sorry

end NUMINAMATH_GPT_roots_ratio_sum_l1046_104698


namespace NUMINAMATH_GPT_sampling_is_systematic_l1046_104645

-- Define the total seats in each row and the total number of rows
def total_seats_per_row : ℕ := 25
def total_rows : ℕ := 30

-- Define a function to identify if the sampling is systematic
def is_systematic_sampling (sample_count : ℕ) (n : ℕ) (interval : ℕ) : Prop :=
  interval = total_seats_per_row ∧ sample_count = total_rows

-- Define the count and interval for the problem
def sample_count : ℕ := 30
def sampling_interval : ℕ := 25

-- Theorem statement: Given the conditions, it is systematic sampling
theorem sampling_is_systematic :
  is_systematic_sampling sample_count total_rows sampling_interval = true :=
sorry

end NUMINAMATH_GPT_sampling_is_systematic_l1046_104645


namespace NUMINAMATH_GPT_new_volume_l1046_104629

variable (l w h : ℝ)

-- Given conditions
def volume := l * w * h = 5000
def surface_area := l * w + l * h + w * h = 975
def sum_of_edges := l + w + h = 60

-- Statement to prove
theorem new_volume (h1 : volume l w h) (h2 : surface_area l w h) (h3 : sum_of_edges l w h) :
  (l + 2) * (w + 2) * (h + 2) = 7198 :=
by
  sorry

end NUMINAMATH_GPT_new_volume_l1046_104629


namespace NUMINAMATH_GPT_distance_between_cities_l1046_104649

theorem distance_between_cities
    (v_bus : ℕ) (v_car : ℕ) (t_bus_meet : ℚ) (t_car_wait : ℚ)
    (d_overtake : ℚ) (s : ℚ)
    (h_vb : v_bus = 40)
    (h_vc : v_car = 50)
    (h_tbm : t_bus_meet = 0.25)
    (h_tcw : t_car_wait = 0.25)
    (h_do : d_overtake = 20)
    (h_eq : (s - 10) / 50 + t_car_wait = (s - 30) / 40) :
    s = 160 :=
by
    exact sorry

end NUMINAMATH_GPT_distance_between_cities_l1046_104649


namespace NUMINAMATH_GPT_strictly_monotone_function_l1046_104664

open Function

-- Define the problem
theorem strictly_monotone_function (f : ℝ → ℝ) (F : ℝ → ℝ → ℝ)
  (hf_cont : Continuous f) (hf_nonconst : ¬ (∃ c, ∀ x, f x = c))
  (hf_eq : ∀ x y : ℝ, f (x + y) = F (f x) (f y)) :
  StrictMono f :=
sorry

end NUMINAMATH_GPT_strictly_monotone_function_l1046_104664


namespace NUMINAMATH_GPT_tom_weekly_fluid_intake_l1046_104627

-- Definitions based on the conditions.
def soda_cans_per_day : ℕ := 5
def ounces_per_can : ℕ := 12
def water_ounces_per_day : ℕ := 64
def days_per_week : ℕ := 7

-- The mathematical proof problem statement.
theorem tom_weekly_fluid_intake :
  (soda_cans_per_day * ounces_per_can + water_ounces_per_day) * days_per_week = 868 := 
by
  sorry

end NUMINAMATH_GPT_tom_weekly_fluid_intake_l1046_104627


namespace NUMINAMATH_GPT_geometric_seq_a4_l1046_104671

theorem geometric_seq_a4 (a : ℕ → ℕ) (q : ℕ) (h_q : q = 2) 
  (h_a1a3 : a 0 * a 2 = 6 * a 1) : a 3 = 24 :=
by
  -- Skipped proof
  sorry

end NUMINAMATH_GPT_geometric_seq_a4_l1046_104671


namespace NUMINAMATH_GPT_number_of_tens_in_sum_l1046_104625

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end NUMINAMATH_GPT_number_of_tens_in_sum_l1046_104625


namespace NUMINAMATH_GPT_no_solution_fraction_eq_l1046_104626

theorem no_solution_fraction_eq (m : ℝ) : 
  ¬(∃ x : ℝ, x ≠ -1 ∧ 3 * x / (x + 1) = m / (x + 1) + 2) ↔ m = -3 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_fraction_eq_l1046_104626


namespace NUMINAMATH_GPT_area_of_shaded_region_l1046_104605

/-- A 4-inch by 4-inch square adjoins a 10-inch by 10-inch square. 
The bottom right corner of the smaller square touches the midpoint of the left side of the larger square. 
Prove that the area of the shaded region is 92/7 square inches. -/
theorem area_of_shaded_region : 
  let small_square_side := 4
  let large_square_side := 10 
  let midpoint := large_square_side / 2
  let height_from_midpoint := midpoint - small_square_side / 2
  let dg := (height_from_midpoint * small_square_side) / ((midpoint + height_from_midpoint))
  (small_square_side * small_square_side) - ((1/2) * dg * small_square_side) = 92 / 7 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1046_104605


namespace NUMINAMATH_GPT_initial_blocks_l1046_104614

theorem initial_blocks (used_blocks remaining_blocks : ℕ) (h1 : used_blocks = 25) (h2 : remaining_blocks = 72) : 
  used_blocks + remaining_blocks = 97 := by
  sorry

end NUMINAMATH_GPT_initial_blocks_l1046_104614


namespace NUMINAMATH_GPT_a_equals_1_or_2_l1046_104610

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x : ℤ | x^2 - 3 * x < 0}
def non_empty_intersection (a : ℤ) : Prop := (M a ∩ N).Nonempty

theorem a_equals_1_or_2 (a : ℤ) (h : non_empty_intersection a) : a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_GPT_a_equals_1_or_2_l1046_104610


namespace NUMINAMATH_GPT_min_distance_sum_l1046_104672

theorem min_distance_sum (x : ℝ) : 
  ∃ y, y = |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| ∧ y = 45 / 8 :=
sorry

end NUMINAMATH_GPT_min_distance_sum_l1046_104672


namespace NUMINAMATH_GPT_green_passes_blue_at_46_l1046_104622

variable {t : ℕ}
variable {k1 k2 k3 k4 : ℝ}
variable {b1 b2 b3 b4 : ℝ}

def elevator_position (k : ℝ) (b : ℝ) (t : ℕ) : ℝ := k * t + b

axiom red_catches_blue_at_36 :
  elevator_position k1 b1 36 = elevator_position k2 b2 36

axiom red_passes_green_at_42 :
  elevator_position k1 b1 42 = elevator_position k3 b3 42

axiom red_passes_yellow_at_48 :
  elevator_position k1 b1 48 = elevator_position k4 b4 48

axiom yellow_passes_blue_at_51 :
  elevator_position k4 b4 51 = elevator_position k2 b2 51

axiom yellow_catches_green_at_54 :
  elevator_position k4 b4 54 = elevator_position k3 b3 54

theorem green_passes_blue_at_46 : 
  elevator_position k3 b3 46 = elevator_position k2 b2 46 := 
sorry

end NUMINAMATH_GPT_green_passes_blue_at_46_l1046_104622


namespace NUMINAMATH_GPT_robin_packages_l1046_104696

theorem robin_packages (p t n : ℕ) (h1 : p = 18) (h2 : t = 486) : t / p = n ↔ n = 27 :=
by
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_robin_packages_l1046_104696


namespace NUMINAMATH_GPT_geometric_sequence_sum_t_value_l1046_104613

theorem geometric_sequence_sum_t_value 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (t : ℝ)
  (h1 : ∀ n : ℕ, S_n n = 3^((n:ℝ)-1) + t)
  (h2 : a_n 1 = 3^0 + t)
  (geometric : ∀ n : ℕ, n ≥ 2 → a_n n = 2 * 3^(n-2)) :
  t = -1/3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_t_value_l1046_104613


namespace NUMINAMATH_GPT_total_peaches_in_baskets_l1046_104668

def total_peaches (red_peaches : ℕ) (green_peaches : ℕ) (baskets : ℕ) : ℕ :=
  (red_peaches + green_peaches) * baskets

theorem total_peaches_in_baskets :
  total_peaches 19 4 15 = 345 :=
by
  sorry

end NUMINAMATH_GPT_total_peaches_in_baskets_l1046_104668


namespace NUMINAMATH_GPT_maximum_value_l1046_104666

theorem maximum_value (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
    (h_eq : a^2 * (b + c - a) = b^2 * (a + c - b) ∧ b^2 * (a + c - b) = c^2 * (b + a - c)) :
    (2 * b + 3 * c) / a = 5 := 
sorry

end NUMINAMATH_GPT_maximum_value_l1046_104666


namespace NUMINAMATH_GPT_avg_stoppage_time_is_20_minutes_l1046_104669

noncomputable def avg_stoppage_time : Real :=
let train1 := (60, 40) -- without stoppages, with stoppages (in kmph)
let train2 := (75, 50) -- without stoppages, with stoppages (in kmph)
let train3 := (90, 60) -- without stoppages, with stoppages (in kmph)
let time1 := (train1.1 - train1.2 : Real) / train1.1
let time2 := (train2.1 - train2.2 : Real) / train2.1
let time3 := (train3.1 - train3.2 : Real) / train3.1
let total_time := time1 + time2 + time3
(total_time / 3) * 60 -- convert hours to minutes

theorem avg_stoppage_time_is_20_minutes :
  avg_stoppage_time = 20 :=
sorry

end NUMINAMATH_GPT_avg_stoppage_time_is_20_minutes_l1046_104669


namespace NUMINAMATH_GPT_find_width_l1046_104685

theorem find_width (A : ℕ) (hA : A ≥ 120) (w : ℕ) (l : ℕ) (hl : l = w + 20) (h_area : w * l = A) : w = 4 :=
by sorry

end NUMINAMATH_GPT_find_width_l1046_104685


namespace NUMINAMATH_GPT_three_digit_integer_one_more_than_multiple_l1046_104656

theorem three_digit_integer_one_more_than_multiple :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n = 841 ∧ ∃ k : ℕ, n = 840 * k + 1 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_integer_one_more_than_multiple_l1046_104656


namespace NUMINAMATH_GPT_total_packs_of_groceries_is_14_l1046_104600

-- Define the number of packs of cookies
def packs_of_cookies : Nat := 2

-- Define the number of packs of cakes
def packs_of_cakes : Nat := 12

-- Define the total packs of groceries as the sum of packs of cookies and cakes
def total_packs_of_groceries : Nat := packs_of_cookies + packs_of_cakes

-- The theorem which states that the total packs of groceries is 14
theorem total_packs_of_groceries_is_14 : total_packs_of_groceries = 14 := by
  -- this is where the proof would go
  sorry

end NUMINAMATH_GPT_total_packs_of_groceries_is_14_l1046_104600


namespace NUMINAMATH_GPT_minimum_value_l1046_104651

noncomputable def minValue (x y : ℝ) : ℝ := (2 / x) + (3 / y)

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 20) : minValue x y = 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1046_104651


namespace NUMINAMATH_GPT_triangle_area_l1046_104655

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- Statement of the theorem
theorem triangle_area : (1 / 2) * |(a.1 * b.2 - a.2 * b.1)| = 4.5 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1046_104655


namespace NUMINAMATH_GPT_a_can_be_any_sign_l1046_104663

theorem a_can_be_any_sign (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b)^2 < (c / d)^2) (hcd : c = -d) : True :=
by
  have := h
  subst hcd
  sorry

end NUMINAMATH_GPT_a_can_be_any_sign_l1046_104663


namespace NUMINAMATH_GPT_intensity_of_replacement_paint_l1046_104689

theorem intensity_of_replacement_paint (f : ℚ) (I_new : ℚ) (I_orig : ℚ) (I_repl : ℚ) :
  f = 2/3 → I_new = 40 → I_orig = 60 → I_repl = (40 - 1/3 * 60) * (3/2) := by
  sorry

end NUMINAMATH_GPT_intensity_of_replacement_paint_l1046_104689


namespace NUMINAMATH_GPT_Alex_has_more_than_200_marbles_on_Monday_of_next_week_l1046_104674

theorem Alex_has_more_than_200_marbles_on_Monday_of_next_week :
  ∃ k : ℕ, k > 0 ∧ 3 * 2^k > 200 ∧ k % 7 = 1 := by
  sorry

end NUMINAMATH_GPT_Alex_has_more_than_200_marbles_on_Monday_of_next_week_l1046_104674


namespace NUMINAMATH_GPT_find_four_numbers_l1046_104611

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7)
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) :
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := 
  by
    sorry

end NUMINAMATH_GPT_find_four_numbers_l1046_104611


namespace NUMINAMATH_GPT_simplest_form_expression_l1046_104609

variable {b : ℝ}

theorem simplest_form_expression (h : b ≠ 1) :
  1 - (1 / (2 + (b / (1 - b)))) = 1 / (2 - b) :=
by
  sorry

end NUMINAMATH_GPT_simplest_form_expression_l1046_104609


namespace NUMINAMATH_GPT_combined_work_days_l1046_104630

theorem combined_work_days (W D : ℕ) (h1: ∀ a b : ℕ, a + b = 4) (h2: (1/6:ℝ) = (1/6:ℝ)) :
  D = 4 :=
by
  sorry

end NUMINAMATH_GPT_combined_work_days_l1046_104630


namespace NUMINAMATH_GPT_find_a_l1046_104643

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 + 2 * a * x - Real.log x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x → x ≤ b → a ≤ y → y ≤ b → x ≤ y → f x ≤ f y

theorem find_a (a : ℝ) :
  is_increasing_on (f a) (1 / 3) 2 → a ≥ 4 / 3 :=
sorry

end NUMINAMATH_GPT_find_a_l1046_104643


namespace NUMINAMATH_GPT_min_value_of_f_inequality_for_a_b_l1046_104641

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 2)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  intro x
  sorry

theorem inequality_for_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : 1/a + 1/b = Real.sqrt 3) : 
  1/a^2 + 2/b^2 ≥ 2 := by
  sorry

end NUMINAMATH_GPT_min_value_of_f_inequality_for_a_b_l1046_104641


namespace NUMINAMATH_GPT_monotonic_increasing_iff_l1046_104640

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + 1 / x

theorem monotonic_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 1 < x → f a x ≥ f a 1) ↔ a ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_iff_l1046_104640


namespace NUMINAMATH_GPT_chromosomal_variations_l1046_104657

-- Define the conditions
def condition1 := "Plants grown from anther culture in vitro."
def condition2 := "Addition or deletion of DNA base pairs on chromosomes."
def condition3 := "Free combination of non-homologous chromosomes."
def condition4 := "Crossing over between non-sister chromatids in a tetrad."
def condition5 := "Cells of a patient with Down syndrome have three copies of chromosome 21."

-- Define a concept of belonging to chromosomal variations
def belongs_to_chromosomal_variations (condition: String) : Prop :=
  condition = condition1 ∨ condition = condition5

-- State the theorem
theorem chromosomal_variations :
  belongs_to_chromosomal_variations condition1 ∧ 
  belongs_to_chromosomal_variations condition5 ∧ 
  ¬ (belongs_to_chromosomal_variations condition2 ∨ 
     belongs_to_chromosomal_variations condition3 ∨ 
     belongs_to_chromosomal_variations condition4) :=
by
  sorry

end NUMINAMATH_GPT_chromosomal_variations_l1046_104657


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1046_104659

theorem solution_set_of_inequality :
  ∀ x : ℝ, 3 * x^2 - 2 * x + 1 > 7 ↔ (x < -2/3 ∨ x > 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1046_104659


namespace NUMINAMATH_GPT_pedestrian_speeds_unique_l1046_104694

variables 
  (x y : ℝ)
  (d : ℝ := 105)  -- Distance between cities
  (t1 : ℝ := 7.5) -- Time for current speeds
  (t2 : ℝ := 105 / 13) -- Time for adjusted speeds

theorem pedestrian_speeds_unique :
  (x + y = 14) →
  (3 * x + y = 14) →
  x = 6 ∧ y = 8 :=
by
  intros h1 h2
  have : 2 * x = 12 :=
    by ring_nf; sorry
  have hx : x = 6 :=
    by linarith
  have hy : y = 8 :=
    by linarith
  exact ⟨hx, hy⟩

end NUMINAMATH_GPT_pedestrian_speeds_unique_l1046_104694


namespace NUMINAMATH_GPT_side_length_S2_l1046_104615

theorem side_length_S2 (r s : ℕ) (h1 : 2 * r + s = 2260) (h2 : 2 * r + 3 * s = 3782) : s = 761 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_side_length_S2_l1046_104615


namespace NUMINAMATH_GPT_area_larger_sphere_l1046_104667

variables {r1 r2 r : ℝ}
variables {A1 A2 : ℝ}

-- Declare constants for the problem
def radius_smaller_sphere : ℝ := 4 -- r1
def radius_larger_sphere : ℝ := 6  -- r2
def radius_ball : ℝ := 1           -- r
def area_smaller_sphere : ℝ := 27  -- A1

-- Given conditions
axiom radius_smaller_sphere_condition : r1 = radius_smaller_sphere
axiom radius_larger_sphere_condition : r2 = radius_larger_sphere
axiom radius_ball_condition : r = radius_ball
axiom area_smaller_sphere_condition : A1 = area_smaller_sphere

-- Statement to be proved
theorem area_larger_sphere :
  r1 = radius_smaller_sphere → r2 = radius_larger_sphere → r = radius_ball → A1 = area_smaller_sphere → A2 = 60.75 :=
by
  intros
  sorry

end NUMINAMATH_GPT_area_larger_sphere_l1046_104667


namespace NUMINAMATH_GPT_select_4_officers_from_7_members_l1046_104686

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Statement of the problem
theorem select_4_officers_from_7_members : binom 7 4 = 35 :=
by
  -- Proof not required, so we use sorry to skip it
  sorry

end NUMINAMATH_GPT_select_4_officers_from_7_members_l1046_104686


namespace NUMINAMATH_GPT_maximum_daily_sales_l1046_104648

def price (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then t + 20
else if (25 ≤ t ∧ t ≤ 30) then -t + 100
else 0

def sales_volume (t : ℕ) : ℝ :=
if (0 < t ∧ t ≤ 30) then -t + 40
else 0

def daily_sales (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then (t + 20) * (-t + 40)
else if (25 ≤ t ∧ t ≤ 30) then (-t + 100) * (-t + 40)
else 0

theorem maximum_daily_sales : ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_sales t = 1125 :=
sorry

end NUMINAMATH_GPT_maximum_daily_sales_l1046_104648


namespace NUMINAMATH_GPT_volume_of_new_pyramid_l1046_104637

theorem volume_of_new_pyramid (l w h : ℝ) (h_vol : (1 / 3) * l * w * h = 80) :
  (1 / 3) * (3 * l) * w * (1.8 * h) = 432 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_new_pyramid_l1046_104637


namespace NUMINAMATH_GPT_more_girls_than_boys_l1046_104688

theorem more_girls_than_boys
  (b g : ℕ)
  (ratio : b / g = 3 / 4)
  (total : b + g = 42) :
  g - b = 6 :=
sorry

end NUMINAMATH_GPT_more_girls_than_boys_l1046_104688


namespace NUMINAMATH_GPT_lcm_12_21_30_l1046_104620

theorem lcm_12_21_30 : Nat.lcm (Nat.lcm 12 21) 30 = 420 := by
  sorry

end NUMINAMATH_GPT_lcm_12_21_30_l1046_104620


namespace NUMINAMATH_GPT_oxygen_atom_diameter_in_scientific_notation_l1046_104691

theorem oxygen_atom_diameter_in_scientific_notation :
  0.000000000148 = 1.48 * 10^(-10) :=
sorry

end NUMINAMATH_GPT_oxygen_atom_diameter_in_scientific_notation_l1046_104691


namespace NUMINAMATH_GPT_distance_from_Idaho_to_Nevada_l1046_104650

theorem distance_from_Idaho_to_Nevada (d1 d2 s1 s2 t total_time : ℝ) 
  (h1 : d1 = 640)
  (h2 : s1 = 80)
  (h3 : s2 = 50)
  (h4 : total_time = 19)
  (h5 : t = total_time - (d1 / s1)) :
  d2 = s2 * t :=
by
  sorry

end NUMINAMATH_GPT_distance_from_Idaho_to_Nevada_l1046_104650


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l1046_104683

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l1046_104683


namespace NUMINAMATH_GPT_remainder_76_pow_77_mod_7_l1046_104693

/-- Statement of the problem:
Prove that the remainder of \(76^{77}\) divided by 7 is 6.
-/
theorem remainder_76_pow_77_mod_7 :
  (76 ^ 77) % 7 = 6 := 
by
  sorry

end NUMINAMATH_GPT_remainder_76_pow_77_mod_7_l1046_104693


namespace NUMINAMATH_GPT_ratio_of_perimeters_l1046_104678

theorem ratio_of_perimeters (L : ℝ) (H : ℝ) (hL1 : L = 8) 
  (hH1 : H = 8) (hH2 : H = 2 * (H / 2)) (hH3 : 4 > 0) (hH4 : 0 < 4 / 3)
  (hW1 : ∀ a, a / 3 > 0 → 8 = L )
  (hPsmall : ∀ P, P = 2 * ((4 / 3) + 8) )
  (hPlarge : ∀ P, P = 2 * ((H - 4 / 3) + 8) )
  :
  (2 * ((4 / 3) + 8)) / (2 * ((8 - (4 / 3)) + 8)) = (7 / 11) := by
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l1046_104678


namespace NUMINAMATH_GPT_tiling_vertex_squares_octagons_l1046_104658

theorem tiling_vertex_squares_octagons (m n : ℕ) 
  (h1 : 135 * n + 90 * m = 360) : 
  m = 1 ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_tiling_vertex_squares_octagons_l1046_104658


namespace NUMINAMATH_GPT_double_average_l1046_104635

theorem double_average (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (h1 : n = 25) (h2 : initial_avg = 70) (h3 : new_avg * n = 2 * (initial_avg * n)) : new_avg = 140 :=
sorry

end NUMINAMATH_GPT_double_average_l1046_104635


namespace NUMINAMATH_GPT_sin_cos_of_theta_l1046_104644

open Real

theorem sin_cos_of_theta (θ : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4))
  (hxθ : ∃ r, r > 0 ∧ P = (r * cos θ, r * sin θ)) :
  sin θ + cos θ = 1 / 5 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_of_theta_l1046_104644


namespace NUMINAMATH_GPT_exists_x_for_log_eqn_l1046_104634

theorem exists_x_for_log_eqn (a : ℝ) (ha : 0 < a) :
  ∃ (x : ℝ), (1 < x) ∧ (Real.log (a * x) / Real.log 10 = 2 * Real.log (x - 1) / Real.log 10) ∧ 
  x = (2 + a + Real.sqrt (a^2 + 4*a)) / 2 := sorry

end NUMINAMATH_GPT_exists_x_for_log_eqn_l1046_104634


namespace NUMINAMATH_GPT_work_done_by_forces_l1046_104695

-- Definitions of given forces and displacement
noncomputable def F1 : ℝ × ℝ := (Real.log 2, Real.log 2)
noncomputable def F2 : ℝ × ℝ := (Real.log 5, Real.log 2)
noncomputable def S : ℝ × ℝ := (2 * Real.log 5, 1)

-- Statement of the theorem
theorem work_done_by_forces :
  let F := (F1.1 + F2.1, F1.2 + F2.2)
  let W := F.1 * S.1 + F.2 * S.2
  W = 2 :=
by
  sorry

end NUMINAMATH_GPT_work_done_by_forces_l1046_104695


namespace NUMINAMATH_GPT_binary_addition_to_hex_l1046_104624

theorem binary_addition_to_hex :
  let n₁ := (0b11111111111 : ℕ)
  let n₂ := (0b11111111 : ℕ)
  n₁ + n₂ = 0x8FE :=
by {
  sorry
}

end NUMINAMATH_GPT_binary_addition_to_hex_l1046_104624


namespace NUMINAMATH_GPT_total_practice_hours_l1046_104699

def schedule : List ℕ := [6, 4, 5, 7, 3]

-- We define the conditions
def total_scheduled_hours : ℕ := schedule.sum

def average_daily_practice_time (total : ℕ) : ℕ := total / schedule.length

def rainy_day_lost_hours : ℕ := average_daily_practice_time total_scheduled_hours

def player_A_missed_hours : ℕ := 2

def player_B_missed_hours : ℕ := 3

def total_missed_hours : ℕ := player_A_missed_hours + player_B_missed_hours

def total_hours_practiced : ℕ := total_scheduled_hours - (rainy_day_lost_hours + total_missed_hours)

-- Now we state the theorem we want to prove
theorem total_practice_hours : total_hours_practiced = 15 := by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_total_practice_hours_l1046_104699


namespace NUMINAMATH_GPT_problem_1_problem_2_l1046_104607

theorem problem_1 :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
by
  sorry

theorem problem_2 :
  (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) = (3^32 - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1046_104607


namespace NUMINAMATH_GPT_exists_distinct_numbers_satisfy_conditions_l1046_104602

theorem exists_distinct_numbers_satisfy_conditions :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a + b + c = 6) ∧
  (2 * b = a + c) ∧
  ((b^2 = a * c) ∨ (a^2 = b * c) ∨ (c^2 = a * b)) :=
by
  sorry

end NUMINAMATH_GPT_exists_distinct_numbers_satisfy_conditions_l1046_104602


namespace NUMINAMATH_GPT_calculate_expression_l1046_104652

theorem calculate_expression : 
  ((13^13 / 13^12)^3 * 3^3) / 3^6 = 27 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1046_104652


namespace NUMINAMATH_GPT_garden_borders_length_l1046_104662

theorem garden_borders_length 
  (a b c d e : ℕ)
  (h1 : 6 * 7 = a^2 + b^2 + c^2 + d^2 + e^2)
  (h2 : a * a + b * b + c * c + d * d + e * e = 42) -- This is analogous to the condition
    
: 15 = (4*a + 4*b + 4*c + 4*d + 4*e - 2*(6 + 7)) / 2 :=
by sorry

end NUMINAMATH_GPT_garden_borders_length_l1046_104662


namespace NUMINAMATH_GPT_investment_at_6_percent_l1046_104676

variables (x y : ℝ)

-- Conditions from the problem
def total_investment : Prop := x + y = 15000
def total_interest : Prop := 0.06 * x + 0.075 * y = 1023

-- Conclusion to prove
def invest_6_percent (x : ℝ) : Prop := x = 6800

theorem investment_at_6_percent (h1 : total_investment x y) (h2 : total_interest x y) : invest_6_percent x :=
by
  sorry

end NUMINAMATH_GPT_investment_at_6_percent_l1046_104676


namespace NUMINAMATH_GPT_police_catches_thief_in_two_hours_l1046_104665

noncomputable def time_to_catch (speed_thief speed_police distance_police_start lead_time : ℝ) : ℝ :=
  let distance_thief := speed_thief * lead_time
  let initial_distance := distance_police_start - distance_thief
  let relative_speed := speed_police - speed_thief
  initial_distance / relative_speed

theorem police_catches_thief_in_two_hours :
  time_to_catch 20 40 60 1 = 2 := by
  sorry

end NUMINAMATH_GPT_police_catches_thief_in_two_hours_l1046_104665


namespace NUMINAMATH_GPT_last_digit_of_4_over_3_power_5_l1046_104612

noncomputable def last_digit_of_fraction (n d : ℕ) : ℕ :=
  (n * 10^5 / d) % 10

def four : ℕ := 4
def three_power_five : ℕ := 3^5

theorem last_digit_of_4_over_3_power_5 :
  last_digit_of_fraction four three_power_five = 7 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_of_4_over_3_power_5_l1046_104612


namespace NUMINAMATH_GPT_trip_duration_l1046_104619

noncomputable def start_time : ℕ := 11 * 60 + 25 -- 11:25 a.m. in minutes
noncomputable def end_time : ℕ := 16 * 60 + 43 + 38 / 60 -- 4:43:38 p.m. in minutes

theorem trip_duration :
  end_time - start_time = 5 * 60 + 18 := 
sorry

end NUMINAMATH_GPT_trip_duration_l1046_104619


namespace NUMINAMATH_GPT_percentage_of_female_students_25_or_older_l1046_104603

theorem percentage_of_female_students_25_or_older
  (T : ℝ) (M F : ℝ) (P : ℝ)
  (h1 : M = 0.40 * T)
  (h2 : F = 0.60 * T)
  (h3 : 0.56 = (0.20 * T) + (0.60 * (1 - P) * T)) :
  P = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_female_students_25_or_older_l1046_104603


namespace NUMINAMATH_GPT_expansion_gameplay_hours_l1046_104638

theorem expansion_gameplay_hours :
  let total_gameplay := 100
  let boring_percentage := 80 / 100
  let enjoyable_percentage := 1 - boring_percentage
  let enjoyable_gameplay_original := enjoyable_percentage * total_gameplay
  let enjoyable_gameplay_total := 50
  let expansion_hours := enjoyable_gameplay_total - enjoyable_gameplay_original
  expansion_hours = 30 :=
by
  let total_gameplay := 100
  let boring_percentage := 80 / 100
  let enjoyable_percentage := 1 - boring_percentage
  let enjoyable_gameplay_original := enjoyable_percentage * total_gameplay
  let enjoyable_gameplay_total := 50
  let expansion_hours := enjoyable_gameplay_total - enjoyable_gameplay_original
  show expansion_hours = 30
  sorry

end NUMINAMATH_GPT_expansion_gameplay_hours_l1046_104638


namespace NUMINAMATH_GPT_contradiction_in_stock_price_l1046_104680

noncomputable def stock_price_contradiction : Prop :=
  ∃ (P D : ℝ), (D = 0.20 * P) ∧ (0.10 = (D / P) * 100)

theorem contradiction_in_stock_price : ¬(stock_price_contradiction) := sorry

end NUMINAMATH_GPT_contradiction_in_stock_price_l1046_104680


namespace NUMINAMATH_GPT_find_a_l1046_104687

-- We define the conditions given in the problem
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The expression defined as per the problem statement
def expansion_coeff_x2 (a : ℝ) : ℝ :=
  (binom 4 2) * 4 - 2 * (binom 4 1) * (binom 5 1) * a + (binom 5 2) * a^2

-- We now express the proof statement in Lean 4. 
-- We need to prove that given the coefficient of x^2 is -16, then a = 2
theorem find_a (a : ℝ) (h : expansion_coeff_x2 a = -16) : a = 2 :=
  by sorry

end NUMINAMATH_GPT_find_a_l1046_104687


namespace NUMINAMATH_GPT_solution_set_inequality_l1046_104692

theorem solution_set_inequality (x : ℝ) : x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 :=
by sorry

end NUMINAMATH_GPT_solution_set_inequality_l1046_104692


namespace NUMINAMATH_GPT_weighted_average_of_angles_l1046_104661

def triangle_inequality (a b c α β γ : ℝ) : Prop :=
  (a - b) * (α - β) ≥ 0 ∧ (b - c) * (β - γ) ≥ 0 ∧ (a - c) * (α - γ) ≥ 0

noncomputable def angle_sum (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi

theorem weighted_average_of_angles (a b c α β γ : ℝ)
  (h1 : triangle_inequality a b c α β γ)
  (h2 : angle_sum α β γ) :
  Real.pi / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_weighted_average_of_angles_l1046_104661


namespace NUMINAMATH_GPT_subletter_payment_correct_l1046_104654

noncomputable def johns_monthly_rent : ℕ := 900
noncomputable def johns_yearly_rent : ℕ := johns_monthly_rent * 12
noncomputable def johns_profit_per_year : ℕ := 3600
noncomputable def total_rent_collected : ℕ := johns_yearly_rent + johns_profit_per_year
noncomputable def number_of_subletters : ℕ := 3
noncomputable def subletter_annual_payment : ℕ := total_rent_collected / number_of_subletters
noncomputable def subletter_monthly_payment : ℕ := subletter_annual_payment / 12

theorem subletter_payment_correct :
  subletter_monthly_payment = 400 :=
by
  sorry

end NUMINAMATH_GPT_subletter_payment_correct_l1046_104654


namespace NUMINAMATH_GPT_probability_at_least_one_two_l1046_104632

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end NUMINAMATH_GPT_probability_at_least_one_two_l1046_104632


namespace NUMINAMATH_GPT_oscar_marathon_training_l1046_104621

theorem oscar_marathon_training :
  let initial_miles := 2
  let target_miles := 20
  let increment_per_week := (2 : ℝ) / 3
  ∃ weeks_required, target_miles - initial_miles = weeks_required * increment_per_week → weeks_required = 27 :=
by
  sorry

end NUMINAMATH_GPT_oscar_marathon_training_l1046_104621


namespace NUMINAMATH_GPT_chord_of_ellipse_bisected_by_point_l1046_104617

theorem chord_of_ellipse_bisected_by_point :
  ∀ (x y : ℝ),
  (∃ (x₁ x₂ y₁ y₂ : ℝ), 
    ( (x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 2) ∧ 
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧ 
    (x₂^2 / 36 + y₂^2 / 9 = 1)) →
  (x + 2 * y = 8) :=
by
  sorry

end NUMINAMATH_GPT_chord_of_ellipse_bisected_by_point_l1046_104617


namespace NUMINAMATH_GPT_total_yield_l1046_104642

theorem total_yield (x y z : ℝ)
  (h1 : 0.4 * z + 0.2 * x = 1)
  (h2 : 0.1 * y - 0.1 * z = -0.5)
  (h3 : 0.1 * x + 0.2 * y = 4) :
  x + y + z = 15 :=
sorry

end NUMINAMATH_GPT_total_yield_l1046_104642


namespace NUMINAMATH_GPT_exists_n_with_common_divisor_l1046_104677

theorem exists_n_with_common_divisor :
  ∃ (n : ℕ), ∀ (k : ℕ), (k ≤ 20) → Nat.gcd (n + k) 30030 > 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_n_with_common_divisor_l1046_104677


namespace NUMINAMATH_GPT_a_4_is_11_l1046_104608

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_4_is_11 : a 4 = 11 := by
  sorry

end NUMINAMATH_GPT_a_4_is_11_l1046_104608


namespace NUMINAMATH_GPT_geometric_series_sum_l1046_104660

theorem geometric_series_sum :
  let a := 1
  let r := (1 / 4 : ℚ)
  (a / (1 - r)) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1046_104660


namespace NUMINAMATH_GPT_opposite_of_neg_eight_l1046_104604

theorem opposite_of_neg_eight (y : ℤ) (h : y + (-8) = 0) : y = 8 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_opposite_of_neg_eight_l1046_104604


namespace NUMINAMATH_GPT_inequality_proof_l1046_104633

theorem inequality_proof (a b c d : ℝ) (h : a > 0) (h : b > 0) (h : c > 0) (h : d > 0)
  (h₁ : (a * b) / (c * d) = (a + b) / (c + d)) : (a + b) * (c + d) ≥ (a + c) * (b + d) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1046_104633


namespace NUMINAMATH_GPT_probability_red_or_white_is_11_over_13_l1046_104684

-- Given data
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - blue_marbles - red_marbles

def blue_size : ℕ := 2
def red_size : ℕ := 1
def white_size : ℕ := 1

-- Total size value of all marbles
def total_size_value : ℕ := (blue_size * blue_marbles) + (red_size * red_marbles) + (white_size * white_marbles)

-- Probability of selecting a red or white marble
def probability_red_or_white : ℚ := (red_size * red_marbles + white_size * white_marbles) / total_size_value

-- Theorem to prove
theorem probability_red_or_white_is_11_over_13 : probability_red_or_white = 11 / 13 :=
by sorry

end NUMINAMATH_GPT_probability_red_or_white_is_11_over_13_l1046_104684


namespace NUMINAMATH_GPT_investment_calculation_l1046_104647

theorem investment_calculation :
  ∃ (x : ℝ), x * (1.04 ^ 14) = 1000 := by
  use 571.75
  sorry

end NUMINAMATH_GPT_investment_calculation_l1046_104647


namespace NUMINAMATH_GPT_employee_discount_percentage_l1046_104636

theorem employee_discount_percentage:
  let purchase_price := 500
  let markup_percentage := 0.15
  let savings := 57.5
  let retail_price := purchase_price * (1 + markup_percentage)
  let discount_percentage := (savings / retail_price) * 100
  discount_percentage = 10 :=
by
  sorry

end NUMINAMATH_GPT_employee_discount_percentage_l1046_104636


namespace NUMINAMATH_GPT_hospital_bed_occupancy_l1046_104646

theorem hospital_bed_occupancy 
  (x : ℕ)
  (beds_A := x)
  (beds_B := 2 * x)
  (beds_C := 3 * x)
  (occupied_A := (1 / 3) * x)
  (occupied_B := (1 / 2) * (2 * x))
  (occupied_C := (1 / 4) * (3 * x))
  (max_capacity_B := (3 / 4) * (2 * x))
  (max_capacity_C := (5 / 6) * (3 * x)) :
  (4 / 3 * x) / (2 * x) = 2 / 3 ∧ (3 / 4 * x) / (3 * x) = 1 / 4 := 
  sorry

end NUMINAMATH_GPT_hospital_bed_occupancy_l1046_104646


namespace NUMINAMATH_GPT_train_length_l1046_104675

variable (L_train : ℝ)
variable (speed_kmhr : ℝ := 45)
variable (time_seconds : ℝ := 30)
variable (bridge_length_m : ℝ := 275)
variable (train_speed_ms : ℝ := speed_kmhr * (1000 / 3600))
variable (total_distance : ℝ := train_speed_ms * time_seconds)

theorem train_length
  (h_total : total_distance = L_train + bridge_length_m) :
  L_train = 100 :=
by 
  sorry

end NUMINAMATH_GPT_train_length_l1046_104675


namespace NUMINAMATH_GPT_cyclists_meet_at_starting_point_l1046_104623

-- Define the conditions: speeds of cyclists and the circumference of the circle
def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def circumference : ℝ := 300

-- Define the total speed by summing individual speeds
def relative_speed : ℝ := speed_cyclist1 + speed_cyclist2

-- Define the time required to meet at the starting point
def meeting_time : ℝ := 20

-- The theorem statement which states that given the conditions, the cyclists will meet after 20 seconds
theorem cyclists_meet_at_starting_point :
  meeting_time = circumference / relative_speed :=
sorry

end NUMINAMATH_GPT_cyclists_meet_at_starting_point_l1046_104623


namespace NUMINAMATH_GPT_find_Y_exists_l1046_104606

variable {X : Finset ℕ} -- Consider a finite set X of natural numbers for generality
variable (S : Finset (Finset ℕ)) -- Set of all subsets of X with even number of elements
variable (f : Finset ℕ → ℝ) -- Real-valued function on subsets of X

-- Conditions
variable (hS : ∀ s ∈ S, s.card % 2 = 0) -- All elements in S have even number of elements
variable (h1 : ∃ A ∈ S, f A > 1990) -- f(A) > 1990 for some A ∈ S
variable (h2 : ∀ ⦃B C⦄, B ∈ S → C ∈ S → (Disjoint B C) → (f (B ∪ C) = f B + f C - 1990)) -- f respects the functional equation for disjoint subsets

theorem find_Y_exists :
  ∃ Y ⊆ X, (∀ D ∈ S, D ⊆ Y → f D > 1990) ∧ (∀ D ∈ S, D ⊆ (X \ Y) → f D ≤ 1990) :=
by
  sorry

end NUMINAMATH_GPT_find_Y_exists_l1046_104606


namespace NUMINAMATH_GPT_area_of_feasible_region_l1046_104631

theorem area_of_feasible_region :
  (∃ k m : ℝ, (∀ x y : ℝ,
    (kx - y + 1 ≥ 0 ∧ kx - my ≤ 0 ∧ y ≥ 0) ↔
    (x - y + 1 ≥ 0 ∧ x + y ≤ 0 ∧ y ≥ 0)) ∧
    k = 1 ∧ m = -1) →
  ∃ a : ℝ, a = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_area_of_feasible_region_l1046_104631


namespace NUMINAMATH_GPT_problem_l1046_104681

theorem problem (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, f (x + 2) = -f x) :
  f 4 = 0 ∧ (∀ x, f (x + 4) = f x) ∧ (∀ x, f (2 - x) = f (2 + x)) :=
sorry

end NUMINAMATH_GPT_problem_l1046_104681


namespace NUMINAMATH_GPT_find_x_l1046_104697

variable (x : ℝ)

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x : delta (phi x) = 23 → x = -1 / 6 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l1046_104697


namespace NUMINAMATH_GPT_find_a_find_min_difference_l1046_104601

noncomputable def f (a x : ℝ) : ℝ := x + a * Real.log x
noncomputable def g (a b x : ℝ) : ℝ := f a x + (1 / 2) * x ^ 2 - b * x

theorem find_a (a : ℝ) (h_perpendicular : (1 : ℝ) + a = 2) : a = 1 := 
sorry

theorem find_min_difference (a b x1 x2 : ℝ) (h_b : b ≥ (7 / 2)) 
    (hx1_lt_hx2 : x1 < x2) (hx_sum : x1 + x2 = b - 1)
    (hx_prod : x1 * x2 = 1) :
    g a b x1 - g a b x2 = (15 / 8) - 2 * Real.log 2 :=
sorry

end NUMINAMATH_GPT_find_a_find_min_difference_l1046_104601
