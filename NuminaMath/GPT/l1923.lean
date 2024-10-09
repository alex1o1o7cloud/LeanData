import Mathlib

namespace inequality_solution_l1923_192392

theorem inequality_solution {x : ℝ} :
  ((x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x)) ↔
  ((x - 2) * (x - 3) * (x - 4) / ((x - 1) * (x - 5) * (x - 6)) > 0) := sorry

end inequality_solution_l1923_192392


namespace find_x_l1923_192316

theorem find_x (x : ℝ) (hx_pos : 0 < x) (h: (x / 100) * x = 4) : x = 20 := by
  sorry

end find_x_l1923_192316


namespace percentage_of_students_absent_l1923_192304

theorem percentage_of_students_absent (total_students : ℕ) (students_present : ℕ) 
(h_total : total_students = 50) (h_present : students_present = 43)
(absent_students := total_students - students_present) :
((absent_students : ℝ) / total_students) * 100 = 14 :=
by sorry

end percentage_of_students_absent_l1923_192304


namespace f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l1923_192361

noncomputable def f (x : ℝ) : ℝ := (1 / 4) ^ x + (1 / 2) ^ x - 1
noncomputable def g (x m : ℝ) : ℝ := (1 - m * 2 ^ x) / (1 + m * 2 ^ x)

theorem f_range_and_boundedness :
  ∀ x : ℝ, x < 0 → 1 < f x ∧ ¬(∃ M : ℝ, ∀ x : ℝ, x < 0 → |f x| ≤ M) :=
by sorry

theorem g_odd_and_bounded (x : ℝ) :
  g x 1 = -g (-x) 1 ∧ |g x 1| < 1 :=
by sorry

theorem g_upper_bound (m : ℝ) (hm : 0 < m ∧ m < 1 / 2) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g x m ≤ (1 - m) / (1 + m) :=
by sorry

end f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l1923_192361


namespace perimeter_of_triangle_l1923_192372

-- Define the average length of the sides of the triangle
def average_length (a b c : ℕ) : ℕ := (a + b + c) / 3

-- Define the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The theorem we want to prove
theorem perimeter_of_triangle {a b c : ℕ} (h_avg : average_length a b c = 12) : perimeter a b c = 36 :=
sorry

end perimeter_of_triangle_l1923_192372


namespace part1_part2_l1923_192377

open Set

def A : Set ℤ := { x | ∃ (m n : ℤ), x = m^2 - n^2 }

theorem part1 : 3 ∈ A := 
by sorry

theorem part2 (k : ℤ) : 4 * k - 2 ∉ A := 
by sorry

end part1_part2_l1923_192377


namespace eq_d_is_quadratic_l1923_192301

def is_quadratic (eq : ℕ → ℤ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ eq 2 = a ∧ eq 1 = b ∧ eq 0 = c

def eq_cond_1 (n : ℕ) : ℤ :=
  match n with
  | 2 => 1  -- x^2 coefficient
  | 1 => 0  -- x coefficient
  | 0 => -1 -- constant term
  | _ => 0

theorem eq_d_is_quadratic : is_quadratic eq_cond_1 :=
  sorry

end eq_d_is_quadratic_l1923_192301


namespace miles_total_instruments_l1923_192324

theorem miles_total_instruments :
  let fingers := 10
  let hands := 2
  let heads := 1
  let trumpets := fingers - 3
  let guitars := hands + 2
  let trombones := heads + 2
  let french_horns := guitars - 1
  (trumpets + guitars + trombones + french_horns) = 17 :=
by
  sorry

end miles_total_instruments_l1923_192324


namespace option_C_correct_l1923_192351

theorem option_C_correct (a b : ℝ) : (-2 * a * b^3)^2 = 4 * a^2 * b^6 :=
by
  sorry

end option_C_correct_l1923_192351


namespace student_l1923_192321

theorem student's_incorrect_answer (D I : ℕ) (h1 : D / 36 = 58) (h2 : D / 87 = I) : I = 24 :=
sorry

end student_l1923_192321


namespace original_price_of_boots_l1923_192393

theorem original_price_of_boots (P : ℝ) (h : P * 0.80 = 72) : P = 90 :=
by 
  sorry

end original_price_of_boots_l1923_192393


namespace volume_of_max_area_rect_prism_l1923_192379

noncomputable def side_length_of_square_base (P: ℕ) : ℕ := P / 4

noncomputable def area_of_square_base (side: ℕ) : ℕ := side * side

noncomputable def volume_of_rectangular_prism (base_area: ℕ) (height: ℕ) : ℕ := base_area * height

theorem volume_of_max_area_rect_prism
  (P : ℕ) (hP : P = 32) 
  (H : ℕ) (hH : H = 9) 
  : volume_of_rectangular_prism (area_of_square_base (side_length_of_square_base P)) H = 576 := 
by
  sorry

end volume_of_max_area_rect_prism_l1923_192379


namespace smallest_portion_is_five_thirds_l1923_192305

theorem smallest_portion_is_five_thirds
    (a1 a2 a3 a4 a5 : ℚ)
    (h1 : a2 = a1 + 1)
    (h2 : a3 = a1 + 2)
    (h3 : a4 = a1 + 3)
    (h4 : a5 = a1 + 4)
    (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
    (h_cond : (1 / 7) * (a3 + a4 + a5) = a1 + a2) :
    a1 = 5 / 3 :=
by
  sorry

end smallest_portion_is_five_thirds_l1923_192305


namespace kennedy_softball_park_miles_l1923_192378

theorem kennedy_softball_park_miles :
  let miles_per_gallon := 19
  let gallons_of_gas := 2
  let total_drivable_miles := miles_per_gallon * gallons_of_gas
  let miles_to_school := 15
  let miles_to_burger_restaurant := 2
  let miles_to_friends_house := 4
  let miles_home := 11
  total_drivable_miles - (miles_to_school + miles_to_burger_restaurant + miles_to_friends_house + miles_home) = 6 :=
by
  sorry

end kennedy_softball_park_miles_l1923_192378


namespace colleen_pencils_l1923_192323

theorem colleen_pencils (joy_pencils : ℕ) (pencil_cost : ℕ) (extra_cost : ℕ) (colleen_paid : ℕ)
  (H1 : joy_pencils = 30)
  (H2 : pencil_cost = 4)
  (H3 : extra_cost = 80)
  (H4 : colleen_paid = (joy_pencils * pencil_cost) + extra_cost) :
  colleen_paid / pencil_cost = 50 := 
by 
  -- Hints, if necessary
sorry

end colleen_pencils_l1923_192323


namespace l_shaped_area_l1923_192395

theorem l_shaped_area (A B C D : Type) (side_abcd: ℝ) (side_small_1: ℝ) (side_small_2: ℝ)
  (area_abcd : side_abcd = 6)
  (area_small_1 : side_small_1 = 2)
  (area_small_2 : side_small_2 = 4)
  (no_overlap : true) :
  side_abcd * side_abcd - (side_small_1 * side_small_1 + side_small_2 * side_small_2) = 16 := by
  sorry

end l_shaped_area_l1923_192395


namespace vasya_improved_example1_vasya_improved_example2_l1923_192363

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l1923_192363


namespace domain_of_c_eq_real_l1923_192330

theorem domain_of_c_eq_real (m : ℝ) : (∀ x : ℝ, m * x^2 - 3 * x + 2 * m ≠ 0) ↔ (m < -3 * Real.sqrt 2 / 4 ∨ m > 3 * Real.sqrt 2 / 4) :=
by
  sorry

end domain_of_c_eq_real_l1923_192330


namespace find_b_l1923_192389

-- Let's define the real numbers and the conditions given.
variables (b y a : ℝ)

-- Conditions from the problem
def condition1 := abs (b - y) = b + y - a
def condition2 := abs (b + y) = b + a

-- The goal is to find the value of b
theorem find_b (h1 : condition1 b y a) (h2 : condition2 b y a) : b = 1 :=
by
  sorry

end find_b_l1923_192389


namespace find_angle_A_find_b_c_l1923_192398
open Real

-- Part I: Proving angle A
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h₁ : (a + b + c) * (b + c - a) = 3 * b * c) :
  A = π / 3 :=
by sorry

-- Part II: Proving values of b and c given a=2 and area of triangle ABC is √3
theorem find_b_c (A B C : ℝ) (a b c : ℝ) (h₁ : a = 2) (h₂ : (1 / 2) * b * c * (sin (π / 3)) = sqrt 3) :
  b = 2 ∧ c = 2 :=
by sorry

end find_angle_A_find_b_c_l1923_192398


namespace carson_giant_slide_rides_l1923_192303

theorem carson_giant_slide_rides :
  let total_hours := 4
  let roller_coaster_wait_time := 30
  let roller_coaster_rides := 4
  let tilt_a_whirl_wait_time := 60
  let tilt_a_whirl_rides := 1
  let giant_slide_wait_time := 15
  -- Convert hours to minutes
  let total_minutes := total_hours * 60
  -- Calculate total wait time for roller coaster
  let roller_coaster_total_wait := roller_coaster_wait_time * roller_coaster_rides
  -- Calculate total wait time for tilt-a-whirl
  let tilt_a_whirl_total_wait := tilt_a_whirl_wait_time * tilt_a_whirl_rides
  -- Calculate total wait time for roller coaster and tilt-a-whirl
  let total_wait := roller_coaster_total_wait + tilt_a_whirl_total_wait
  -- Calculate remaining time
  let remaining_time := total_minutes - total_wait
  -- Calculate how many times Carson can ride the giant slide
  let giant_slide_rides := remaining_time / giant_slide_wait_time
  giant_slide_rides = 4 := by
  let total_hours := 4
  let roller_coaster_wait_time := 30
  let roller_coaster_rides := 4
  let tilt_a_whirl_wait_time := 60
  let tilt_a_whirl_rides := 1
  let giant_slide_wait_time := 15
  let total_minutes := total_hours * 60
  let roller_coaster_total_wait := roller_coaster_wait_time * roller_coaster_rides
  let tilt_a_whirl_total_wait := tilt_a_whirl_wait_time * tilt_a_whirl_rides
  let total_wait := roller_coaster_total_wait + tilt_a_whirl_total_wait
  let remaining_time := total_minutes - total_wait
  let giant_slide_rides := remaining_time / giant_slide_wait_time
  show giant_slide_rides = 4
  sorry

end carson_giant_slide_rides_l1923_192303


namespace terry_lunch_combos_l1923_192309

def num_lettuce : ℕ := 2
def num_tomatoes : ℕ := 3
def num_olives : ℕ := 4
def num_soups : ℕ := 2

theorem terry_lunch_combos : num_lettuce * num_tomatoes * num_olives * num_soups = 48 :=
by
  sorry

end terry_lunch_combos_l1923_192309


namespace proof_problem_l1923_192356

variable (y θ Q : ℝ)

-- Given condition
def condition : Prop := 5 * (3 * y + 7 * Real.sin θ) = Q

-- Goal to be proved
def goal : Prop := 15 * (9 * y + 21 * Real.sin θ) = 9 * Q

theorem proof_problem (h : condition y θ Q) : goal y θ Q :=
by
  sorry

end proof_problem_l1923_192356


namespace gcd_polynomial_is_25_l1923_192338

theorem gcd_polynomial_is_25 (b : ℕ) (h : ∃ k : ℕ, b = 2700 * k) :
  Nat.gcd (b^2 + 27 * b + 75) (b + 25) = 25 :=
by 
    sorry

end gcd_polynomial_is_25_l1923_192338


namespace mailman_junk_mail_l1923_192328

variable (junk_mail_per_house : ℕ) (houses_per_block : ℕ)

theorem mailman_junk_mail (h1 : junk_mail_per_house = 2) (h2 : houses_per_block = 7) :
  junk_mail_per_house * houses_per_block = 14 :=
by
  sorry

end mailman_junk_mail_l1923_192328


namespace pipe_fills_tank_without_leak_l1923_192349

theorem pipe_fills_tank_without_leak (T : ℝ) (h1 : 1 / 6 = 1 / T - 1 / 12) : T = 4 :=
by
  sorry

end pipe_fills_tank_without_leak_l1923_192349


namespace sarah_score_l1923_192346

-- Given conditions
variable (s g : ℕ) -- Sarah's score and Greg's score
variable (h1 : s = g + 60) -- Sarah's score is 60 points more than Greg's
variable (h2 : (s + g) / 2 = 130) -- The average of their two scores is 130

-- Proof statement
theorem sarah_score : s = 160 :=
by
  sorry

end sarah_score_l1923_192346


namespace slope_intercept_of_line_l1923_192355

theorem slope_intercept_of_line :
  ∃ (l : ℝ → ℝ), (∀ x, l x = (4 * x - 9) / 3) ∧ l 3 = 1 ∧ ∃ k, k / (1 + k^2) = 1 / 2 ∧ l x = (k^2 - 1) / (1 + k^2) := sorry

end slope_intercept_of_line_l1923_192355


namespace smallest_number_l1923_192333

theorem smallest_number (x y z : ℕ) (h1 : y = 4 * x) (h2 : z = 2 * y) 
(h3 : (x + y + z) / 3 = 78) : x = 18 := 
by 
    sorry

end smallest_number_l1923_192333


namespace candies_left_is_correct_l1923_192343

-- Define the number of candies bought on different days
def candiesBoughtTuesday : ℕ := 3
def candiesBoughtThursday : ℕ := 5
def candiesBoughtFriday : ℕ := 2

-- Define the number of candies eaten
def candiesEaten : ℕ := 6

-- Define the total candies left
def candiesLeft : ℕ := (candiesBoughtTuesday + candiesBoughtThursday + candiesBoughtFriday) - candiesEaten

theorem candies_left_is_correct : candiesLeft = 4 := by
  -- Placeholder proof: replace 'sorry' with the actual proof when necessary
  sorry

end candies_left_is_correct_l1923_192343


namespace randy_mango_trees_l1923_192327

theorem randy_mango_trees (M C : ℕ) 
  (h1 : C = M / 2 - 5) 
  (h2 : M + C = 85) : 
  M = 60 := 
sorry

end randy_mango_trees_l1923_192327


namespace pizza_eaten_after_six_trips_l1923_192344

theorem pizza_eaten_after_six_trips :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 729 :=
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  have : S_n = (1 / 3) * (1 - (1 / 3)^6) / (1 - 1 / 3) := by sorry
  have : S_n = 364 / 729 := by sorry
  exact this

end pizza_eaten_after_six_trips_l1923_192344


namespace find_number_l1923_192339

-- Definitions based on the given conditions
def area (s : ℝ) := s^2
def perimeter (s : ℝ) := 4 * s
def given_perimeter : ℝ := 36
def equation (s : ℝ) (n : ℝ) := 5 * area s = 10 * perimeter s + n

-- Statement of the problem
theorem find_number :
  ∃ n : ℝ, equation (given_perimeter / 4) n ∧ n = 45 :=
by
  sorry

end find_number_l1923_192339


namespace z_is_1_2_decades_younger_than_x_l1923_192367

variable (X Y Z : ℝ)

theorem z_is_1_2_decades_younger_than_x (h : X + Y = Y + Z + 12) : (X - Z) / 10 = 1.2 :=
by
  sorry

end z_is_1_2_decades_younger_than_x_l1923_192367


namespace train_crosses_pole_l1923_192385

theorem train_crosses_pole
  (speed_kmph : ℝ)
  (train_length_meters : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (time_seconds : ℝ)
  (h1 : speed_kmph = 270)
  (h2 : train_length_meters = 375.03)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_mps = speed_kmph * conversion_factor)
  (h5 : time_seconds = train_length_meters / speed_mps)
  : time_seconds = 5.0004 :=
by
  sorry

end train_crosses_pole_l1923_192385


namespace mary_cut_10_roses_l1923_192326

-- Define the initial and final number of roses
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses cut as the difference between final and initial
def roses_cut : ℕ :=
  final_roses - initial_roses

-- Theorem stating the number of roses cut by Mary
theorem mary_cut_10_roses : roses_cut = 10 := by
  sorry

end mary_cut_10_roses_l1923_192326


namespace prism_pyramid_fusion_l1923_192331

theorem prism_pyramid_fusion :
  ∃ (result_faces result_edges result_vertices : ℕ),
    result_faces + result_edges + result_vertices = 28 ∧
    ((result_faces = 8 ∧ result_edges = 13 ∧ result_vertices = 7) ∨
    (result_faces = 7 ∧ result_edges = 12 ∧ result_vertices = 7)) :=
by
  sorry

end prism_pyramid_fusion_l1923_192331


namespace domain_of_f_l1923_192350

def domain_f (x : ℝ) : Prop := x ≤ 4 ∧ x ≠ 1

theorem domain_of_f :
  {x : ℝ | ∃(h1 : 4 - x ≥ 0) (h2 : x - 1 ≠ 0), true} = {x : ℝ | domain_f x} :=
by
  sorry

end domain_of_f_l1923_192350


namespace g_inequality_solution_range_of_m_l1923_192376

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 8
noncomputable def g (x : ℝ) : ℝ := 2*x^2 - 4*x - 16
noncomputable def h (x m : ℝ) : ℝ := x^2 - (4 + m)*x + (m + 7)

theorem g_inequality_solution:
  {x : ℝ | g x < 0} = {x : ℝ | -2 < x ∧ x < 4} :=
by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 1 → f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 4 :=
by
  sorry

end g_inequality_solution_range_of_m_l1923_192376


namespace prime_between_30_and_40_with_remainder_1_l1923_192362

theorem prime_between_30_and_40_with_remainder_1 (n : ℕ) : 
  n.Prime → 
  30 < n → n < 40 → 
  n % 6 = 1 → 
  n = 37 := 
sorry

end prime_between_30_and_40_with_remainder_1_l1923_192362


namespace infinite_representable_and_nonrepresentable_terms_l1923_192307

def a (n : ℕ) : ℕ :=
  2^n + 2^(n / 2)

def is_representable (k : ℕ) : Prop :=   
  -- A nonnegative integer is defined to be representable if it can
  -- be expressed as a sum of distinct terms from the sequence a(n).
  sorry  -- Definition will depend on the specific notion of representability

theorem infinite_representable_and_nonrepresentable_terms :
  (∃ᶠ n in at_top, is_representable (a n)) ∧ (∃ᶠ n in at_top, ¬is_representable (a n)) :=
sorry  -- This is the main theorem claiming infinitely many representable and non-representable terms.

end infinite_representable_and_nonrepresentable_terms_l1923_192307


namespace square_traffic_sign_perimeter_l1923_192366

-- Define the side length of the square
def side_length : ℕ := 4

-- Define the number of sides of the square
def number_of_sides : ℕ := 4

-- Define the perimeter of the square
def perimeter (l : ℕ) (n : ℕ) : ℕ := l * n

-- The theorem to be proved
theorem square_traffic_sign_perimeter : perimeter side_length number_of_sides = 16 :=
by
  sorry

end square_traffic_sign_perimeter_l1923_192366


namespace inequality_proof_l1923_192387

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := 
by
  sorry

end inequality_proof_l1923_192387


namespace find_k_range_l1923_192332

noncomputable def f (k x : ℝ) : ℝ := (k * x + 1 / 3) * Real.exp x - x

theorem find_k_range : 
  (∃ (k : ℝ), ∀ (x : ℕ), x > 0 → (f k (x : ℝ) < 0 ↔ x = 1)) ↔
  (k ≥ 1 / (Real.exp 2) - 1 / 6 ∧ k < 1 / Real.exp 1 - 1 / 3) :=
sorry

end find_k_range_l1923_192332


namespace problem_l1923_192369

-- Define proposition p: for all x in ℝ, x^2 + 1 ≥ 1
def p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

-- Define proposition q: for angles A and B in a triangle, A > B ↔ sin A > sin B
def q : Prop := ∀ {A B : ℝ}, A > B ↔ Real.sin A > Real.sin B

-- The problem definition: prove that p ∨ q is true
theorem problem (hp : p) (hq : q) : p ∨ q := sorry

end problem_l1923_192369


namespace range_of_m_l1923_192336

theorem range_of_m (m : ℝ) (p : |m + 1| ≤ 2) (q : ¬(m^2 - 4 ≥ 0)) : -2 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l1923_192336


namespace range_of_m_l1923_192382

def y1 (m x : ℝ) : ℝ :=
  m * (x - 2 * m) * (x + m + 2)

def y2 (x : ℝ) : ℝ :=
  x - 1

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, y1 m x < 0 ∨ y2 x < 0) ∧ (∃ x : ℝ, x < -3 ∧ y1 m x * y2 x < 0) ↔ (-4 < m ∧ m < -3/2) := 
by
  sorry

end range_of_m_l1923_192382


namespace population_30_3_million_is_30300000_l1923_192386

theorem population_30_3_million_is_30300000 :
  let million := 1000000
  let population_1998 := 30.3 * million
  population_1998 = 30300000 :=
by
  -- Proof goes here
  sorry

end population_30_3_million_is_30300000_l1923_192386


namespace ship_cargo_weight_l1923_192347

theorem ship_cargo_weight (initial_cargo_tons additional_cargo_tons : ℝ) (unloaded_cargo_pounds : ℝ)
    (ton_to_kg pound_to_kg : ℝ) :
    initial_cargo_tons = 5973.42 →
    additional_cargo_tons = 8723.18 →
    unloaded_cargo_pounds = 2256719.55 →
    ton_to_kg = 907.18474 →
    pound_to_kg = 0.45359237 →
    (initial_cargo_tons * ton_to_kg + additional_cargo_tons * ton_to_kg - unloaded_cargo_pounds * pound_to_kg = 12302024.7688159) :=
by
  intros
  sorry

end ship_cargo_weight_l1923_192347


namespace binary_multiplication_l1923_192317

/-- 
Calculate the product of two binary numbers and validate the result.
Given:
  a = 1101 in base 2,
  b = 111 in base 2,
Prove:
  a * b = 1011110 in base 2. 
-/
theorem binary_multiplication : 
  let a := 0b1101
  let b := 0b111
  a * b = 0b1011110 :=
by
  sorry

end binary_multiplication_l1923_192317


namespace sqrt_inequality_l1923_192360

theorem sqrt_inequality (a b c : ℝ) (θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * (Real.cos θ)^2 + b * (Real.sin θ)^2 < c) :
  Real.sqrt a * (Real.cos θ)^2 + Real.sqrt b * (Real.sin θ)^2 < Real.sqrt c :=
sorry

end sqrt_inequality_l1923_192360


namespace ellipse_foci_coordinates_l1923_192310

theorem ellipse_foci_coordinates :
  (∀ (x y : ℝ), (x^2 / 16 + y^2 / 25 = 1) → (∃ (c : ℝ), c = 3 ∧ (x = 0 ∧ (y = c ∨ y = -c)))) :=
by
  sorry

end ellipse_foci_coordinates_l1923_192310


namespace smallest_possible_integer_l1923_192308

theorem smallest_possible_integer (a b : ℤ)
  (a_lt_10 : a < 10)
  (b_lt_10 : b < 10)
  (a_lt_b : a < b)
  (sum_eq_45 : a + b + 32 = 45)
  : a = 4 :=
by
  sorry

end smallest_possible_integer_l1923_192308


namespace ants_no_collision_probability_l1923_192300

-- Definitions
def cube_vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

def adjacent (v : ℕ) : Finset ℕ :=
  match v with
  | 0 => {1, 3, 4}
  | 1 => {0, 2, 5}
  | 2 => {1, 3, 6}
  | 3 => {0, 2, 7}
  | 4 => {0, 5, 7}
  | 5 => {1, 4, 6}
  | 6 => {2, 5, 7}
  | 7 => {3, 4, 6}
  | _ => ∅

-- Hypothesis: Each ant moves independently to one of the three adjacent vertices.

-- Result to prove
def X : ℕ := sorry  -- The number of valid ways ants can move without collisions

theorem ants_no_collision_probability : 
  ∃ X, (X / (3 : ℕ)^8 = X / 6561) :=
  by
    sorry

end ants_no_collision_probability_l1923_192300


namespace new_area_is_497_l1923_192370

noncomputable def rect_area_proof : Prop :=
  ∃ (l w l' w' : ℝ),
    -- initial area condition
    l * w = 540 ∧ 
    -- conditions for new dimensions
    l' = 0.8 * l ∧
    w' = 1.15 * w ∧
    -- final area calculation
    l' * w' = 497

theorem new_area_is_497 : rect_area_proof := by
  sorry

end new_area_is_497_l1923_192370


namespace parabola_standard_eq_line_m_tangent_l1923_192329

open Real

variables (p k : ℝ) (x y : ℝ)

-- Definitions based on conditions
def parabola_equation (p : ℝ) : Prop := ∀ x y : ℝ, x^2 = 2 * p * y
def line_m (k : ℝ) : Prop := ∀ x y : ℝ, y = k * x + 6

-- Problem statement
theorem parabola_standard_eq (p : ℝ) (hp : p = 2) :
  parabola_equation p ↔ (∀ x y : ℝ, x^2 = 4 * y) :=
sorry

theorem line_m_tangent (k : ℝ) (x1 x2 : ℝ)
  (hpq : x1 + x2 = 4 * k ∧ x1 * x2 = -24)
  (hk : k = 1/2 ∨ k = -1/2) :
  line_m k ↔ ((k = 1/2 ∧ ∀ x y : ℝ, y = 1/2 * x + 6) ∨ (k = -1/2 ∧ ∀ x y : ℝ, y = -1/2 * x + 6)) :=
sorry

end parabola_standard_eq_line_m_tangent_l1923_192329


namespace arithmetic_sequence_fifth_term_l1923_192391

theorem arithmetic_sequence_fifth_term :
  ∀ (a d : ℤ), (a + 19 * d = 15) → (a + 20 * d = 18) → (a + 4 * d = -30) :=
by
  intros a d h1 h2
  sorry

end arithmetic_sequence_fifth_term_l1923_192391


namespace least_possible_value_of_smallest_integer_l1923_192337

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), 
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A + B + C + D) / 4 = 68 →
    D = 90 →
    A = 5 :=
by
  intros A B C D h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end least_possible_value_of_smallest_integer_l1923_192337


namespace largest_unsatisfiable_group_l1923_192315

theorem largest_unsatisfiable_group :
  ∃ n : ℕ, (∀ a b c : ℕ, n ≠ 6 * a + 9 * b + 20 * c) ∧ (∀ m : ℕ, m > n → ∃ a b c : ℕ, m = 6 * a + 9 * b + 20 * c) ∧ n = 43 :=
by
  sorry

end largest_unsatisfiable_group_l1923_192315


namespace trip_time_l1923_192334

theorem trip_time (x : ℝ) (T : ℝ) :
  (70 * 4 + 60 * 5 + 50 * x) / (4 + 5 + x) = 58 → 
  T = 4 + 5 + x → 
  T = 16.25 :=
by
  intro h1 h2
  sorry

end trip_time_l1923_192334


namespace min_value_a_plus_b_l1923_192368

theorem min_value_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : a^2 ≥ 8 * b) (h4 : b^2 ≥ a) : a + b ≥ 6 := by
  sorry

end min_value_a_plus_b_l1923_192368


namespace perfect_squares_from_equation_l1923_192375

theorem perfect_squares_from_equation (x y : ℕ) (h : 2 * x^2 + x = 3 * y^2 + y) :
  ∃ a b c : ℕ, x - y = a^2 ∧ 2 * x + 2 * y + 1 = b^2 ∧ 3 * x + 3 * y + 1 = c^2 :=
by
  sorry

end perfect_squares_from_equation_l1923_192375


namespace part1_part2_l1923_192354

variables (q x : ℝ)
def f (x : ℝ) (q : ℝ) : ℝ := x^2 - 16*x + q + 3
def g (x : ℝ) (q : ℝ) : ℝ := f x q + 51

theorem part1 (h1 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x q = 0):
  (-20 : ℝ) ≤ q ∧ q ≤ 12 := 
  sorry

theorem part2 (h2 : ∀ x ∈ Set.Icc (q : ℝ) 10, g x q ≥ 0) : 
  9 ≤ q ∧ q < 10 := 
  sorry

end part1_part2_l1923_192354


namespace simplify_expression_l1923_192381

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : (x^2)⁻¹ - 2 = (1 - 2 * x^2) / (x^2) :=
by
  -- proof here
  sorry

end simplify_expression_l1923_192381


namespace find_certain_number_l1923_192359

theorem find_certain_number (x : ℕ) (h: x - 82 = 17) : x = 99 :=
by
  sorry

end find_certain_number_l1923_192359


namespace compare_star_l1923_192384

def star (m n : ℤ) : ℤ := (m + 2) * 3 - n

theorem compare_star : star 2 (-2) > star (-2) 2 := 
by sorry

end compare_star_l1923_192384


namespace find_initial_sum_l1923_192348

-- Define the conditions as constants
def A1 : ℝ := 590
def A2 : ℝ := 815
def t1 : ℝ := 2
def t2 : ℝ := 7

-- Define the variables
variable (P r : ℝ)

-- First condition after 2 years
def condition1 : Prop := A1 = P + P * r * t1

-- Second condition after 7 years
def condition2 : Prop := A2 = P + P * r * t2

-- The statement we need to prove: the initial sum of money P is 500
theorem find_initial_sum (h1 : condition1 P r) (h2 : condition2 P r) : P = 500 :=
sorry

end find_initial_sum_l1923_192348


namespace per_capita_income_growth_l1923_192399

theorem per_capita_income_growth (x : ℝ) : 
  (250 : ℝ) * (1 + x) ^ 20 ≥ 800 →
  (250 : ℝ) * (1 + x) ^ 40 ≥ 2560 := 
by
  intros h
  -- Proof is not required, so we skip it with sorry
  sorry

end per_capita_income_growth_l1923_192399


namespace find_m_l1923_192365

theorem find_m (x y m : ℝ)
  (h1 : 6 * x + 3 = 0)
  (h2 : 3 * y + m = 15)
  (h3 : x * y = 1) : m = 21 := 
sorry

end find_m_l1923_192365


namespace first_quadrant_sin_cos_inequality_l1923_192374

def is_first_quadrant_angle (α : ℝ) : Prop :=
  0 < Real.sin α ∧ 0 < Real.cos α

theorem first_quadrant_sin_cos_inequality (α : ℝ) :
  (is_first_quadrant_angle α ↔ Real.sin α + Real.cos α > 1) :=
by
  sorry

end first_quadrant_sin_cos_inequality_l1923_192374


namespace product_of_distinct_roots_l1923_192373

theorem product_of_distinct_roots (x1 x2 : ℝ) (hx1 : x1 ^ 2 - 2 * x1 = 1) (hx2 : x2 ^ 2 - 2 * x2 = 1) (h_distinct : x1 ≠ x2) : 
  x1 * x2 = -1 := 
  sorry

end product_of_distinct_roots_l1923_192373


namespace ratio_of_flowers_given_l1923_192322

-- Definitions based on conditions
def Collin_flowers : ℕ := 25
def Ingrid_flowers_initial : ℕ := 33
def petals_per_flower : ℕ := 4
def Collin_petals_total : ℕ := 144

-- The ratio of the number of flowers Ingrid gave to Collin to the number of flowers Ingrid had initially
theorem ratio_of_flowers_given :
  let Ingrid_flowers_given := (Collin_petals_total - (Collin_flowers * petals_per_flower)) / petals_per_flower
  let ratio := Ingrid_flowers_given / Ingrid_flowers_initial
  ratio = 1 / 3 :=
by
  sorry

end ratio_of_flowers_given_l1923_192322


namespace solve_p_l1923_192358

theorem solve_p (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 2 * p + 5 * q = 8) : 
  p = 11 / 19 :=
by
  sorry

end solve_p_l1923_192358


namespace tv_height_l1923_192390

theorem tv_height (H : ℝ) : 
  672 / (24 * H) = (1152 / (48 * 32)) + 1 → 
  H = 16 := 
by
  have h_area_first_TV : 24 * H ≠ 0 := sorry
  have h_new_condition: 1152 / (48 * 32) + 1 = 1.75 := sorry
  have h_cost_condition: 672 / (24 * H) = 1.75 := sorry
  sorry

end tv_height_l1923_192390


namespace arithmetic_sequence_tenth_term_l1923_192394

theorem arithmetic_sequence_tenth_term (a d : ℤ) 
  (h1 : a + 2 * d = 23) 
  (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
by 
  -- proof goes here
  sorry

end arithmetic_sequence_tenth_term_l1923_192394


namespace y_intercepts_count_l1923_192397

theorem y_intercepts_count : 
  ∀ (a b c : ℝ), a = 3 ∧ b = (-4) ∧ c = 5 → (b^2 - 4*a*c < 0) → ∀ y : ℝ, x = 3*y^2 - 4*y + 5 → x ≠ 0 :=
by
  sorry

end y_intercepts_count_l1923_192397


namespace computer_price_increase_l1923_192311

theorem computer_price_increase (c : ℕ) (h : 2 * c = 540) : c + (c * 30 / 100) = 351 :=
by
  sorry

end computer_price_increase_l1923_192311


namespace problem_1_problem_2_l1923_192345

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x - 2)

theorem problem_1 (a b : ℝ) (h1 : f a b 3 - 3 + 12 = 0) (h2 : f a b 4 - 4 + 12 = 0) :
  f a b x = (2 - x) / (x - 2) := sorry

theorem problem_2 (k : ℝ) (h : k > 1) :
  ∀ x, f (-1) 2 x < k ↔ (if 1 < k ∧ k < 2 then (1 < x ∧ x < k) ∨ (2 < x) 
                         else if k = 2 then 1 < x ∧ x ≠ 2 
                         else (1 < x ∧ x < 2) ∨ (k < x)) := sorry

-- Function definition for clarity
noncomputable def f_spec (x : ℝ) : ℝ := (2 - x) / (x - 2)

end problem_1_problem_2_l1923_192345


namespace linear_function_l1923_192340

theorem linear_function (f : ℝ → ℝ)
  (h : ∀ x, f (f x) = 4 * x + 6) :
  (∀ x, f x = 2 * x + 2) ∨ (∀ x, f x = -2 * x - 6) :=
sorry

end linear_function_l1923_192340


namespace actual_price_of_food_before_tax_and_tip_l1923_192353

theorem actual_price_of_food_before_tax_and_tip 
  (total_paid : ℝ)
  (tip_percentage : ℝ)
  (tax_percentage : ℝ)
  (pre_tax_food_price : ℝ)
  (h1 : total_paid = 132)
  (h2 : tip_percentage = 0.20)
  (h3 : tax_percentage = 0.10)
  (h4 : total_paid = (1 + tip_percentage) * (1 + tax_percentage) * pre_tax_food_price) :
  pre_tax_food_price = 100 :=
by sorry

end actual_price_of_food_before_tax_and_tip_l1923_192353


namespace ellipse_hyperbola_foci_l1923_192313

theorem ellipse_hyperbola_foci (c d : ℝ) 
  (h_ellipse : d^2 - c^2 = 25) 
  (h_hyperbola : c^2 + d^2 = 64) : |c * d| = Real.sqrt 868.5 := by
  sorry

end ellipse_hyperbola_foci_l1923_192313


namespace reporters_percentage_l1923_192335

theorem reporters_percentage (total_reporters : ℕ) (local_politics_percentage : ℝ) (non_politics_percentage : ℝ) :
  local_politics_percentage = 28 → non_politics_percentage = 60 → 
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  100 * (non_local_political_reporters / political_reporters) = 30 :=
by
  intros
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  sorry

end reporters_percentage_l1923_192335


namespace perpendicular_bisector_eq_l1923_192383

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 2)

-- Theorem stating that the perpendicular bisector has the specified equation
theorem perpendicular_bisector_eq : ∀ (x y : ℝ), (y = -2 * x + 3) ↔ ∃ (a b : ℝ), (a, b) = A ∨ (a, b) = B ∧ (y = -2 * x + 3) :=
by
  sorry

end perpendicular_bisector_eq_l1923_192383


namespace otherWorkStations_accommodate_students_l1923_192352

def numTotalStudents := 38
def numStations := 16
def numWorkStationsForTwo := 10
def capacityWorkStationsForTwo := 2

theorem otherWorkStations_accommodate_students : 
  (numTotalStudents - numWorkStationsForTwo * capacityWorkStationsForTwo) = 18 := 
by
  sorry

end otherWorkStations_accommodate_students_l1923_192352


namespace survey_respondents_l1923_192312

theorem survey_respondents (X Y : ℕ) (hX : X = 150) (ratio : X = 5 * Y) : X + Y = 180 :=
by
  sorry

end survey_respondents_l1923_192312


namespace hyperbola_midpoint_l1923_192302

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l1923_192302


namespace geom_seq_q_eq_l1923_192325

theorem geom_seq_q_eq (a1 : ℕ := 2) (S3 : ℕ := 26) 
  (h1 : a1 = 2) 
  (h2 : S3 = 26) : 
  ∃ q : ℝ, (q = 3 ∨ q = -4) := by
  sorry

end geom_seq_q_eq_l1923_192325


namespace total_jumps_correct_l1923_192314

-- Define Ronald's jumps
def Ronald_jumps : ℕ := 157

-- Define the difference in jumps between Rupert and Ronald
def difference : ℕ := 86

-- Define Rupert's jumps
def Rupert_jumps : ℕ := Ronald_jumps + difference

-- Define the total number of jumps
def total_jumps : ℕ := Ronald_jumps + Rupert_jumps

-- State the main theorem we want to prove
theorem total_jumps_correct : total_jumps = 400 := 
by sorry

end total_jumps_correct_l1923_192314


namespace arithmetic_sequence_sum_false_statement_l1923_192306

theorem arithmetic_sequence_sum_false_statement (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n n.succ - a_n n = a_n 1 - a_n 0)
  (h_S : ∀ n, S n = (n + 1) * a_n 0 + (n * (n + 1) * (a_n 1 - a_n 0)) / 2)
  (h1 : S 6 < S 7) (h2 : S 7 = S 8) (h3 : S 8 > S 9) : ¬ (S 10 > S 6) :=
by
  sorry

end arithmetic_sequence_sum_false_statement_l1923_192306


namespace volume_of_rectangular_prism_l1923_192380

theorem volume_of_rectangular_prism
  (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 15)
  (h3 : z * x = 12) :
  x * y * z = 60 :=
sorry

end volume_of_rectangular_prism_l1923_192380


namespace not_all_on_C_implies_exists_not_on_C_l1923_192318

def F (x y : ℝ) : Prop := sorry  -- Define F according to specifics
def on_curve_C (x y : ℝ) : Prop := sorry -- Define what it means to be on curve C according to specifics

theorem not_all_on_C_implies_exists_not_on_C (h : ¬ ∀ x y : ℝ, F x y → on_curve_C x y) :
  ∃ x y : ℝ, F x y ∧ ¬ on_curve_C x y := sorry

end not_all_on_C_implies_exists_not_on_C_l1923_192318


namespace AB_complete_work_together_in_10_days_l1923_192342

-- Definitions for the work rates
def rate_A (work : ℕ) : ℚ := work / 14 -- A's rate of work (work per day)
def rate_AB (work : ℕ) : ℚ := work / 10 -- A and B together's rate of work (work per day)

-- Definition for B's rate of work derived from the combined rate and A's rate
def rate_B (work : ℕ) : ℚ := rate_AB work - rate_A work

-- Definition of the fact that the combined rate should equal their individual rates summed
def combined_rate_equals_sum (work : ℕ) : Prop := rate_AB work = (rate_A work + rate_B work)

-- Statement we need to prove:
theorem AB_complete_work_together_in_10_days (work : ℕ) (h : combined_rate_equals_sum work) : rate_AB work = work / 10 :=
by {
  -- Given conditions are implicitly used without a formal proof here.
  -- To prove that A and B together can indeed complete the work in 10 days.
  sorry
}


end AB_complete_work_together_in_10_days_l1923_192342


namespace kopeechka_items_l1923_192388

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l1923_192388


namespace sally_initial_orange_balloons_l1923_192320

variable (initial_orange_balloons : ℕ)  -- The initial number of orange balloons Sally had
variable (lost_orange_balloons : ℕ := 2)  -- The number of orange balloons Sally lost
variable (current_orange_balloons : ℕ := 7)  -- The number of orange balloons Sally currently has

theorem sally_initial_orange_balloons : 
  current_orange_balloons + lost_orange_balloons = initial_orange_balloons := 
by
  sorry

end sally_initial_orange_balloons_l1923_192320


namespace factorize_expression_l1923_192371

theorem factorize_expression (a b : ℝ) : a^2 + a * b = a * (a + b) := 
by
  sorry

end factorize_expression_l1923_192371


namespace minimum_value_correct_l1923_192396

noncomputable def minimum_value (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a + 3*b = 1 then 1/a + 1/b else 0

theorem minimum_value_correct : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ minimum_value a b = 4 + 4 * Real.sqrt 3 :=
by
  sorry

end minimum_value_correct_l1923_192396


namespace count_of_numbers_with_digit_3_eq_71_l1923_192341

-- Define the problem space
def count_numbers_without_digit_3 : ℕ := 729
def total_numbers : ℕ := 800
def count_numbers_with_digit_3 : ℕ := total_numbers - count_numbers_without_digit_3

-- Prove that the count of numbers from 1 to 800 containing at least one digit 3 is 71
theorem count_of_numbers_with_digit_3_eq_71 :
  count_numbers_with_digit_3 = 71 :=
by
  sorry

end count_of_numbers_with_digit_3_eq_71_l1923_192341


namespace range_of_x_l1923_192364

theorem range_of_x (x : ℝ) : 
  (∀ (m : ℝ), |m| ≤ 1 → x^2 - 2 > m * x) ↔ (x < -2 ∨ x > 2) :=
by 
  sorry

end range_of_x_l1923_192364


namespace find_sixth_number_l1923_192319

theorem find_sixth_number 
  (A : ℕ → ℝ)
  (h1 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 11 = 60))
  (h2 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6) / 6 = 58))
  (h3 : ((A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 6 = 65)) 
  : A 6 = 78 :=
by
  sorry

end find_sixth_number_l1923_192319


namespace curve_crosses_itself_l1923_192357

-- Definitions of the parametric equations
def x (t : ℝ) : ℝ := t^2 - 4
def y (t : ℝ) : ℝ := t^3 - 6*t + 3

-- The theorem statement
theorem curve_crosses_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁, y t₁) = (2, 3) :=
by
  -- Proof would go here
  sorry

end curve_crosses_itself_l1923_192357
