import Mathlib

namespace bread_pieces_theorem_l1550_155003

/-- Number of pieces after tearing a slice of bread in half twice -/
def pieces_per_slice : ℕ := 4

/-- Number of initial bread slices -/
def initial_slices : ℕ := 2

/-- Total number of bread pieces after tearing -/
def total_pieces : ℕ := initial_slices * pieces_per_slice

theorem bread_pieces_theorem : total_pieces = 8 := by
  sorry

end bread_pieces_theorem_l1550_155003


namespace no_positive_integer_solution_l1550_155096

theorem no_positive_integer_solution :
  ¬∃ (x y z t : ℕ+), x^2 + 2*y^2 = z^2 ∧ 2*x^2 + y^2 = t^2 := by
  sorry

end no_positive_integer_solution_l1550_155096


namespace valid_pairs_l1550_155075

def is_valid_pair (x y : ℕ+) : Prop :=
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / Nat.lcm x y + (1 : ℚ) / Nat.gcd x y = (1 : ℚ) / 2

theorem valid_pairs : 
  ∀ x y : ℕ+, is_valid_pair x y ↔ 
    ((x = 5 ∧ y = 20) ∨ 
     (x = 6 ∧ y = 12) ∨ 
     (x = 8 ∧ y = 8) ∨ 
     (x = 8 ∧ y = 12) ∨ 
     (x = 9 ∧ y = 24) ∨ 
     (x = 12 ∧ y = 15) ∨
     (y = 5 ∧ x = 20) ∨ 
     (y = 6 ∧ x = 12) ∨ 
     (y = 8 ∧ x = 12) ∨ 
     (y = 9 ∧ x = 24) ∨ 
     (y = 12 ∧ x = 15)) :=
by sorry

end valid_pairs_l1550_155075


namespace tan_theta_eq_seven_l1550_155027

theorem tan_theta_eq_seven (θ : Real) 
  (h1 : θ > π/4 ∧ θ < π/2) 
  (h2 : Real.cos (θ - π/4) = 4/5) : 
  Real.tan θ = 7 := by
  sorry

end tan_theta_eq_seven_l1550_155027


namespace rectangle_perimeter_and_area_l1550_155031

/-- Perimeter and area of a rectangle with specific dimensions -/
theorem rectangle_perimeter_and_area :
  let l : ℝ := Real.sqrt 6 + 2 * Real.sqrt 5
  let w : ℝ := 2 * Real.sqrt 6 - Real.sqrt 5
  let perimeter : ℝ := 2 * (l + w)
  let area : ℝ := l * w
  (perimeter = 6 * Real.sqrt 6 + 2 * Real.sqrt 5) ∧
  (area = 2 + 3 * Real.sqrt 30) := by
  sorry


end rectangle_perimeter_and_area_l1550_155031


namespace angle_edc_measure_l1550_155091

theorem angle_edc_measure (y : ℝ) :
  let angle_bde : ℝ := 4 * y
  let angle_edc : ℝ := 3 * y
  angle_bde + angle_edc = 180 →
  angle_edc = 540 / 7 := by
sorry

end angle_edc_measure_l1550_155091


namespace S_divisible_by_4003_l1550_155045

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def S : ℕ := factorial 2001 + (List.range 2001).foldl (λ acc i => acc * (2002 + i)) 1

theorem S_divisible_by_4003 : S % 4003 = 0 := by
  sorry

end S_divisible_by_4003_l1550_155045


namespace square_with_triangles_removed_l1550_155071

theorem square_with_triangles_removed (s x y : ℝ) 
  (h1 : s - 2*x = 15)
  (h2 : s - 2*y = 9)
  (h3 : x > 0)
  (h4 : y > 0) :
  4 * (1/2 * x * y) = 67.5 := by
  sorry

end square_with_triangles_removed_l1550_155071


namespace seventh_term_largest_implies_n_l1550_155061

/-- The binomial coefficient -/
def binomial_coefficient (n k : ℕ) : ℕ := sorry

/-- Predicate to check if the 7th term has the largest binomial coefficient -/
def seventh_term_largest (n : ℕ) : Prop :=
  ∀ k, k ≠ 6 → binomial_coefficient n 6 ≥ binomial_coefficient n k

/-- Theorem stating the possible values of n when the 7th term has the largest binomial coefficient -/
theorem seventh_term_largest_implies_n (n : ℕ) :
  seventh_term_largest n → n = 11 ∨ n = 12 ∨ n = 13 := by sorry

end seventh_term_largest_implies_n_l1550_155061


namespace horner_v₂_equals_4_l1550_155094

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 1 + x + x^2 + x^3 + 2x^4 -/
def f : List ℝ := [1, 1, 1, 1, 2]

/-- v₂ in Horner's method for f(x) at x = 1 -/
def v₂ : ℝ :=
  let v₁ := 2 * 1 + 1  -- a₄x + a₃
  v₁ * 1 + 1           -- v₁x + a₂

theorem horner_v₂_equals_4 :
  v₂ = 4 := by sorry

end horner_v₂_equals_4_l1550_155094


namespace final_stamp_count_l1550_155093

def parkers_stamps (initial_stamps : ℕ) (addies_stamps : ℕ) : ℕ :=
  initial_stamps + (addies_stamps / 4)

theorem final_stamp_count : parkers_stamps 18 72 = 36 := by
  sorry

end final_stamp_count_l1550_155093


namespace rectangle_area_l1550_155020

/-- Given a rectangle with length three times its width and diagonal y, prove its area is 3y²/10 -/
theorem rectangle_area (y : ℝ) (y_pos : y > 0) : 
  ∃ w : ℝ, w > 0 ∧ 
  w^2 + (3*w)^2 = y^2 ∧ 
  3 * w^2 = (3 * y^2) / 10 := by
sorry

end rectangle_area_l1550_155020


namespace vegan_menu_fraction_l1550_155079

theorem vegan_menu_fraction (vegan_dishes : ℕ) (total_dishes : ℕ) (soy_dishes : ℕ) :
  vegan_dishes = 6 →
  vegan_dishes = total_dishes / 3 →
  soy_dishes = 4 →
  (vegan_dishes - soy_dishes : ℚ) / total_dishes = 1 / 9 := by
  sorry

end vegan_menu_fraction_l1550_155079


namespace division_by_fraction_problem_solution_l1550_155036

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b := by
  sorry

theorem problem_solution :
  (5 : ℚ) / ((8 : ℚ) / 15) = 75 / 8 := by
  sorry

end division_by_fraction_problem_solution_l1550_155036


namespace savings_calculation_l1550_155086

/-- Calculates the amount saved given sales, basic salary, commission rate, and savings rate -/
def calculate_savings (sales : ℝ) (basic_salary : ℝ) (commission_rate : ℝ) (savings_rate : ℝ) : ℝ :=
  let total_earnings := basic_salary + sales * commission_rate
  total_earnings * savings_rate

/-- Proves that given the specified conditions, the amount saved is $29 -/
theorem savings_calculation :
  let sales := 2500
  let basic_salary := 240
  let commission_rate := 0.02
  let savings_rate := 0.10
  calculate_savings sales basic_salary commission_rate savings_rate = 29 := by
sorry

#eval calculate_savings 2500 240 0.02 0.10

end savings_calculation_l1550_155086


namespace roots_of_polynomial_l1550_155077

theorem roots_of_polynomial (x : ℝ) :
  let p : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x - 3)*(x + 2)
  {x : ℝ | p x = 0} = {2, 3, -2} := by
sorry

end roots_of_polynomial_l1550_155077


namespace correct_calculation_l1550_155074

theorem correct_calculation (n m : ℝ) : n * m^2 - 2 * m^2 * n = -m^2 * n := by
  sorry

end correct_calculation_l1550_155074


namespace chord_ratio_l1550_155032

-- Define the circle and points
variable (circle : Type) (A B C D E P : circle)

-- Define the distance function
variable (dist : circle → circle → ℝ)

-- State the theorem
theorem chord_ratio (h1 : dist A P = 5)
                    (h2 : dist C P = 9)
                    (h3 : dist D E = 4) :
  dist B P / dist E P = 81 / 805 := by sorry

end chord_ratio_l1550_155032


namespace fraction_comparison_l1550_155019

theorem fraction_comparison 
  (a b c d : ℤ) 
  (hc : c ≠ 0) 
  (hd : d ≠ 0) : 
  (c = d ∧ a > b → (a : ℚ) / c > (b : ℚ) / d) ∧
  (a = b ∧ c < d → (a : ℚ) / c > (b : ℚ) / d) ∧
  (a > b ∧ c < d → (a : ℚ) / c > (b : ℚ) / d) :=
by sorry

end fraction_comparison_l1550_155019


namespace roots_sum_of_squares_l1550_155007

theorem roots_sum_of_squares (α β : ℝ) : 
  (α^2 - 2*α - 1 = 0) → (β^2 - 2*β - 1 = 0) → α^2 + β^2 = 6 := by
  sorry

end roots_sum_of_squares_l1550_155007


namespace cistern_fill_time_l1550_155051

def fill_time (rate_a rate_b rate_c : ℚ) : ℚ :=
  1 / (rate_a + rate_b + rate_c)

theorem cistern_fill_time :
  let rate_a : ℚ := 1 / 10
  let rate_b : ℚ := 1 / 12
  let rate_c : ℚ := -1 / 15
  fill_time rate_a rate_b rate_c = 60 / 7 :=
by sorry

end cistern_fill_time_l1550_155051


namespace repeating_decimal_equals_fraction_l1550_155097

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 0.36363636

/-- The fraction 40/99 -/
def fraction : ℚ := 40 / 99

/-- Theorem stating that the repeating decimal 0.363636... equals 40/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l1550_155097


namespace certain_number_subtraction_l1550_155060

theorem certain_number_subtraction (x : ℤ) : 
  (3005 - x + 10 = 2705) → (x = 310) := by
  sorry

end certain_number_subtraction_l1550_155060


namespace crayons_difference_l1550_155030

def birthday_crayons : ℕ := 8597
def crayons_given : ℕ := 7255
def crayons_lost : ℕ := 3689

theorem crayons_difference : crayons_given - crayons_lost = 3566 := by
  sorry

end crayons_difference_l1550_155030


namespace white_surface_fraction_of_given_cube_l1550_155053

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def white_surface_fraction (c : CompositeCube) : ℚ :=
  -- Implementation details omitted
  0

/-- Theorem stating the fraction of white surface area for the given cube -/
theorem white_surface_fraction_of_given_cube :
  let c : CompositeCube := {
    edge_length := 4,
    small_cube_count := 64,
    white_cube_count := 44,
    black_cube_count := 20
  }
  white_surface_fraction c = 5/6 :=
by sorry

end white_surface_fraction_of_given_cube_l1550_155053


namespace no_integer_roots_for_odd_coeff_quadratic_l1550_155078

/-- A quadratic function with odd coefficients has no integer roots -/
theorem no_integer_roots_for_odd_coeff_quadratic (a b c : ℤ) (ha : a ≠ 0) 
  (hodd : Odd a ∧ Odd b ∧ Odd c) :
  ¬∃ x : ℤ, a * x^2 + b * x + c = 0 := by
  sorry

end no_integer_roots_for_odd_coeff_quadratic_l1550_155078


namespace positive_roots_of_x_power_x_l1550_155026

theorem positive_roots_of_x_power_x (x : ℝ) : 
  x > 0 → (x^x = 1 / Real.sqrt 2 ↔ x = 1/2 ∨ x = 1/4) := by
  sorry

end positive_roots_of_x_power_x_l1550_155026


namespace birds_in_tree_l1550_155024

theorem birds_in_tree (initial_birds : ℕ) (new_birds : ℕ) (total_birds : ℕ) :
  initial_birds = 14 →
  new_birds = 21 →
  total_birds = initial_birds + new_birds →
  total_birds = 35 := by
sorry

end birds_in_tree_l1550_155024


namespace test_scores_sum_l1550_155034

/-- Given the scores of Bill, John, and Sue on a test, prove that their total sum is 160 points. -/
theorem test_scores_sum (bill john sue : ℕ) : 
  bill = john + 20 →   -- Bill scored 20 more points than John
  bill = sue / 2 →     -- Bill scored half as many points as Sue
  bill = 45 →          -- Bill received 45 points
  bill + john + sue = 160 := by
sorry

end test_scores_sum_l1550_155034


namespace triangle_angle_measure_l1550_155035

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 4√3, b = 12, and B = 60°, then A = 30° -/
theorem triangle_angle_measure (a b c A B C : ℝ) : 
  a = 4 * Real.sqrt 3 → 
  b = 12 → 
  B = 60 * π / 180 → 
  A = 30 * π / 180 := by sorry

end triangle_angle_measure_l1550_155035


namespace johns_wife_notebooks_l1550_155017

/-- Proves the number of notebooks John's wife bought for each child --/
theorem johns_wife_notebooks (num_children : ℕ) (johns_notebooks_per_child : ℕ) (total_notebooks : ℕ) :
  num_children = 3 →
  johns_notebooks_per_child = 2 →
  total_notebooks = 21 →
  (total_notebooks - num_children * johns_notebooks_per_child) / num_children = 5 := by
sorry

end johns_wife_notebooks_l1550_155017


namespace dartboard_sector_angle_l1550_155025

theorem dartboard_sector_angle (probability : ℝ) (angle : ℝ) : 
  probability = 1/4 → angle = 90 :=
by
  sorry

end dartboard_sector_angle_l1550_155025


namespace run_time_around_square_field_l1550_155048

/-- Calculates the time taken for a boy to run around a square field -/
theorem run_time_around_square_field (side_length : ℝ) (speed_kmh : ℝ) : 
  side_length = 60 → speed_kmh = 9 → 
  (4 * side_length) / (speed_kmh * 1000 / 3600) = 96 := by
  sorry

#check run_time_around_square_field

end run_time_around_square_field_l1550_155048


namespace number_of_girls_in_school_l1550_155006

/-- Given a school with more girls than boys, calculate the number of girls. -/
theorem number_of_girls_in_school 
  (total_pupils : ℕ) 
  (girl_boy_difference : ℕ) 
  (h1 : total_pupils = 926)
  (h2 : girl_boy_difference = 458) :
  ∃ (girls boys : ℕ), 
    girls = boys + girl_boy_difference ∧ 
    girls + boys = total_pupils ∧ 
    girls = 692 := by
  sorry

end number_of_girls_in_school_l1550_155006


namespace waste_fraction_for_park_l1550_155087

/-- A kite-shaped park with specific properties -/
structure KitePark where
  -- AB and BC lengths
  side_length : ℝ
  -- Ensure side_length is positive
  side_positive : side_length > 0

/-- The fraction of the park's area from which waste is brought to the longest diagonal -/
noncomputable def waste_fraction (park : KitePark) : ℝ :=
  7071 / 10000

/-- Theorem stating the waste fraction for a kite park with side length 100 -/
theorem waste_fraction_for_park (park : KitePark) 
  (h : park.side_length = 100) : 
  waste_fraction park = 7071 / 10000 :=
by sorry

end waste_fraction_for_park_l1550_155087


namespace find_b_l1550_155058

theorem find_b (a b c : ℕ) 
  (h1 : 1 < a) (h2 : a < b) (h3 : b < c)
  (h4 : a + b + c = 111)
  (h5 : b^2 = a * c) :
  b = 36 := by
  sorry

end find_b_l1550_155058


namespace samantha_sleep_hours_l1550_155055

/-- Represents a time of day in 24-hour format -/
structure TimeOfDay where
  hour : Nat
  minute : Nat
  is_valid : hour < 24 ∧ minute < 60

/-- Calculates the number of hours between two times -/
def hoursBetween (t1 t2 : TimeOfDay) : Nat :=
  if t2.hour ≥ t1.hour then
    t2.hour - t1.hour
  else
    24 + t2.hour - t1.hour

/-- Samantha's bedtime -/
def bedtime : TimeOfDay := {
  hour := 19,
  minute := 0,
  is_valid := by simp
}

/-- Samantha's wake-up time -/
def wakeupTime : TimeOfDay := {
  hour := 11,
  minute := 0,
  is_valid := by simp
}

theorem samantha_sleep_hours :
  hoursBetween bedtime wakeupTime = 16 := by sorry

end samantha_sleep_hours_l1550_155055


namespace dennis_rocks_theorem_l1550_155005

/-- Calculates the number of rocks Dennis made the fish spit out -/
def rocks_spit_out (initial_rocks : ℕ) (eaten_rocks : ℕ) (final_rocks : ℕ) : ℕ :=
  final_rocks - (initial_rocks - eaten_rocks)

/-- Proves that Dennis made the fish spit out 2 rocks -/
theorem dennis_rocks_theorem (initial_rocks eaten_rocks final_rocks : ℕ) 
  (h1 : initial_rocks = 10)
  (h2 : eaten_rocks = initial_rocks / 2)
  (h3 : final_rocks = 7) :
  rocks_spit_out initial_rocks eaten_rocks final_rocks = 2 := by
sorry

end dennis_rocks_theorem_l1550_155005


namespace robins_initial_gum_pieces_robins_initial_gum_pieces_proof_l1550_155049

/-- Given that Robin now has 62 pieces of gum after receiving 44.0 pieces from her brother,
    prove that her initial number of gum pieces was 18. -/
theorem robins_initial_gum_pieces : ℝ → Prop :=
  fun initial_gum =>
    initial_gum + 44.0 = 62 →
    initial_gum = 18
    
/-- Proof of the theorem -/
theorem robins_initial_gum_pieces_proof : robins_initial_gum_pieces 18 := by
  sorry

end robins_initial_gum_pieces_robins_initial_gum_pieces_proof_l1550_155049


namespace rectangle_cut_and_rearrange_l1550_155052

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the result of cutting and rearranging a rectangle -/
structure CutAndRearrange where
  original : Rectangle
  new : Rectangle

/-- Defines the properties of a valid cut and rearrange operation -/
def isValidCutAndRearrange (cr : CutAndRearrange) : Prop :=
  cr.original.width * cr.original.height = cr.new.width * cr.new.height ∧
  cr.new.width ≠ cr.original.width ∧
  cr.new.height ≠ cr.original.height ∧
  (cr.new.width > cr.new.height → cr.new.width > cr.original.width ∧ cr.new.width > cr.original.height) ∧
  (cr.new.height > cr.new.width → cr.new.height > cr.original.width ∧ cr.new.height > cr.original.height)

/-- The main theorem to be proved -/
theorem rectangle_cut_and_rearrange :
  ∀ (cr : CutAndRearrange),
    cr.original.width = 9 ∧
    cr.original.height = 16 ∧
    isValidCutAndRearrange cr →
    max cr.new.width cr.new.height = 18 := by
  sorry

end rectangle_cut_and_rearrange_l1550_155052


namespace complex_division_problem_l1550_155065

theorem complex_division_problem (z : ℂ) (h : z = 4 + 3*I) : 
  Complex.abs z / z = 4/5 - 3/5*I := by
  sorry

end complex_division_problem_l1550_155065


namespace rectangle_length_proof_l1550_155016

-- Define the rectangle's properties
def rectangle_area : ℝ := 54.3
def rectangle_width : ℝ := 6

-- Theorem statement
theorem rectangle_length_proof :
  let length := rectangle_area / rectangle_width
  length = 9.05 := by
  sorry

end rectangle_length_proof_l1550_155016


namespace largest_prime_factor_of_7999999999_l1550_155076

theorem largest_prime_factor_of_7999999999 :
  let n : ℕ := 7999999999
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q) →
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Prime q → q ∣ n → q ≤ p) ∧
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ p = 4002001) :=
by sorry

end largest_prime_factor_of_7999999999_l1550_155076


namespace sum_of_digits_doubled_l1550_155021

/-- Sum of digits function -/
def S (k : ℕ+) : ℕ := sorry

/-- All digits less than or equal to 7 -/
def digits_le_7 (n : ℕ+) : Prop := sorry

theorem sum_of_digits_doubled (k : ℕ+) :
  S k = 2187 → digits_le_7 (2 * k) → S (2 * k) = 4374 := by sorry

end sum_of_digits_doubled_l1550_155021


namespace sum_of_roots_l1550_155063

theorem sum_of_roots (a b : ℝ) 
  (ha : a^4 - 16*a^3 + 40*a^2 - 50*a + 25 = 0)
  (hb : b^4 - 24*b^3 + 216*b^2 - 720*b + 625 = 0) :
  a + b = 7 ∨ a + b = 3 := by
sorry

end sum_of_roots_l1550_155063


namespace min_words_to_pass_l1550_155067

-- Define the exam parameters
def total_words : ℕ := 800
def passing_score : ℚ := 90 / 100
def guess_rate : ℚ := 10 / 100

-- Define the function to calculate the score based on words learned
def exam_score (words_learned : ℕ) : ℚ :=
  (words_learned : ℚ) / total_words + 
  guess_rate * ((total_words - words_learned) : ℚ) / total_words

-- Theorem statement
theorem min_words_to_pass : 
  ∀ n : ℕ, n < 712 → exam_score n < passing_score ∧ 
  exam_score 712 ≥ passing_score := by sorry

end min_words_to_pass_l1550_155067


namespace clarence_oranges_l1550_155056

/-- The number of oranges Clarence had initially -/
def initial_oranges : ℕ := sorry

/-- The number of oranges Clarence received from Joyce -/
def oranges_from_joyce : ℕ := 3

/-- The total number of oranges Clarence has after receiving oranges from Joyce -/
def total_oranges : ℕ := 8

/-- Theorem stating that the initial number of oranges plus those from Joyce equals the total -/
theorem clarence_oranges : initial_oranges + oranges_from_joyce = total_oranges := by sorry

end clarence_oranges_l1550_155056


namespace exam_maximum_marks_l1550_155066

theorem exam_maximum_marks :
  ∀ (passing_threshold : ℝ) (obtained_marks : ℝ) (failing_margin : ℝ),
    passing_threshold = 0.30 →
    obtained_marks = 30 →
    failing_margin = 36 →
    ∃ (max_marks : ℝ),
      max_marks = 220 ∧
      passing_threshold * max_marks = obtained_marks + failing_margin :=
by sorry

end exam_maximum_marks_l1550_155066


namespace allocation_ways_l1550_155039

-- Define the number of doctors and nurses
def num_doctors : ℕ := 2
def num_nurses : ℕ := 4

-- Define the number of schools
def num_schools : ℕ := 2

-- Define the number of doctors and nurses needed per school
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

-- Theorem statement
theorem allocation_ways :
  (Nat.choose num_doctors doctors_per_school) * (Nat.choose num_nurses nurses_per_school) = 12 :=
sorry

end allocation_ways_l1550_155039


namespace article_cost_l1550_155092

/-- The cost of an article given specific profit conditions -/
theorem article_cost (C : ℝ) (S : ℝ) : 
  S = 1.25 * C → -- Original selling price (25% profit)
  (0.8 * C + 0.3 * (0.8 * C) = S - 6.3) → -- New cost and selling price with 30% profit
  C = 30 := by
sorry

end article_cost_l1550_155092


namespace angle_B_measure_max_perimeter_max_perimeter_achieved_l1550_155073

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.B + t.b * Real.cos t.C + t.c * Real.cos t.B = 0

-- Part I: Prove that angle B is 2π/3
theorem angle_B_measure (t : Triangle) (h : condition t) : t.B = 2 * Real.pi / 3 := by
  sorry

-- Part II: Prove the maximum perimeter
theorem max_perimeter (t : Triangle) (h : condition t) (hb : t.b = Real.sqrt 3) :
  t.a + t.b + t.c ≤ Real.sqrt 3 + 2 := by
  sorry

-- Prove that the maximum perimeter is achieved
theorem max_perimeter_achieved : ∃ (t : Triangle), condition t ∧ t.b = Real.sqrt 3 ∧ t.a + t.b + t.c = Real.sqrt 3 + 2 := by
  sorry

end angle_B_measure_max_perimeter_max_perimeter_achieved_l1550_155073


namespace min_value_reciprocal_sum_l1550_155090

theorem min_value_reciprocal_sum (m n : ℝ) : 
  m > 0 → n > 0 → m * 1 + n * 1 = 2 → (1 / m + 1 / n) ≥ 2 := by
  sorry

end min_value_reciprocal_sum_l1550_155090


namespace number_equation_proof_l1550_155010

theorem number_equation_proof : ∃ x : ℝ, x - (1004 / 20.08) = 4970 ∧ x = 5020 := by
  sorry

end number_equation_proof_l1550_155010


namespace power_two_congruence_l1550_155022

theorem power_two_congruence (n : ℕ) (a : ℤ) (hn : n ≥ 1) (ha : Odd a) :
  a ^ (2 ^ n) ≡ 1 [ZMOD (2 ^ (n + 2))] := by
  sorry

end power_two_congruence_l1550_155022


namespace investment_growth_l1550_155013

/-- Computes the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The problem statement -/
theorem investment_growth : 
  let principal : ℝ := 3000
  let rate : ℝ := 0.07
  let years : ℕ := 25
  ⌊compound_interest principal rate years⌋ = 16281 := by
  sorry

end investment_growth_l1550_155013


namespace arithmetic_sequence_common_difference_l1550_155004

/-- An arithmetic sequence with first term 2 and 10th term 20 has common difference 2 -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term is 2
  a 10 = 20 →                          -- 10th term is 20
  a 2 - a 1 = 2 :=                     -- common difference is 2
by
  sorry

end arithmetic_sequence_common_difference_l1550_155004


namespace min_value_with_product_constraint_l1550_155012

theorem min_value_with_product_constraint (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (product_constraint : x * y * z = 32) : 
  x^2 + 4*x*y + 4*y^2 + 2*z^2 ≥ 68 ∧ 
  (x^2 + 4*x*y + 4*y^2 + 2*z^2 = 68 ↔ x = 4 ∧ y = 2 ∧ z = 4) :=
by sorry

end min_value_with_product_constraint_l1550_155012


namespace train_length_l1550_155009

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 50 → time_s = 9 → 
  (speed_kmh * 1000 / 3600) * time_s = 125 := by sorry

end train_length_l1550_155009


namespace log_equality_implies_value_l1550_155037

theorem log_equality_implies_value (p q : ℝ) (c : ℝ) (h : 0 < p ∧ 0 < q ∧ 0 < 5) :
  Real.log p / Real.log 5 = c - Real.log q / Real.log 5 → p = 5^c / q := by
  sorry

end log_equality_implies_value_l1550_155037


namespace function_extrema_implies_interval_bounds_l1550_155099

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the theorem
theorem function_extrema_implies_interval_bounds
  (a : ℝ)
  (h_nonneg : 0 ≤ a)
  (h_max : ∀ x ∈ Set.Icc 0 a, f x ≤ 3)
  (h_min : ∀ x ∈ Set.Icc 0 a, 2 ≤ f x)
  (h_max_achieved : ∃ x ∈ Set.Icc 0 a, f x = 3)
  (h_min_achieved : ∃ x ∈ Set.Icc 0 a, f x = 2) :
  a ∈ Set.Icc 1 2 :=
sorry

end function_extrema_implies_interval_bounds_l1550_155099


namespace smallest_n_for_square_not_cube_l1550_155062

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y * y = x

def expression (n k : ℕ) : ℕ :=
  3^k + n^k + (3*n)^k + 2014^k

theorem smallest_n_for_square_not_cube :
  ∃ n : ℕ, n > 0 ∧
    (∀ k : ℕ, is_perfect_square (expression n k)) ∧
    (∀ k : ℕ, ¬ is_perfect_cube (expression n k)) ∧
    (∀ m : ℕ, m > 0 ∧ m < n →
      ¬(∀ k : ℕ, is_perfect_square (expression m k)) ∨
      ¬(∀ k : ℕ, ¬ is_perfect_cube (expression m k))) ∧
    n = 2 :=
  sorry

end smallest_n_for_square_not_cube_l1550_155062


namespace kamal_average_marks_l1550_155038

/-- Calculates the average marks given a list of obtained marks and total marks -/
def averageMarks (obtained : List ℕ) (total : List ℕ) : ℚ :=
  (obtained.sum : ℚ) / (total.sum : ℚ) * 100

theorem kamal_average_marks : 
  let obtained := [76, 60, 82, 67, 85, 78]
  let total := [120, 110, 100, 90, 100, 95]
  averageMarks obtained total = 448 / 615 * 100 := by
  sorry

#eval (448 : ℚ) / 615 * 100

end kamal_average_marks_l1550_155038


namespace boat_speed_in_still_water_l1550_155054

/-- The speed of a boat in still water, given its travel distances with and against a stream. -/
theorem boat_speed_in_still_water 
  (along_stream : ℝ) 
  (against_stream : ℝ) 
  (h1 : along_stream = 11) 
  (h2 : against_stream = 7) : 
  ∃ (boat_speed stream_speed : ℝ), 
    boat_speed + stream_speed = along_stream ∧ 
    boat_speed - stream_speed = against_stream ∧ 
    boat_speed = 9 := by
  sorry

end boat_speed_in_still_water_l1550_155054


namespace tan_equality_periodic_l1550_155023

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 → 
  Real.tan (n * π / 180) = Real.tan (1540 * π / 180) → 
  n = 40 := by
sorry

end tan_equality_periodic_l1550_155023


namespace shooting_training_probabilities_l1550_155000

/-- Shooting training probabilities -/
structure ShootingProbabilities where
  nine_or_above : ℝ
  eight_to_nine : ℝ
  seven_to_eight : ℝ
  six_to_seven : ℝ

/-- Theorem for shooting training probabilities -/
theorem shooting_training_probabilities
  (probs : ShootingProbabilities)
  (h1 : probs.nine_or_above = 0.18)
  (h2 : probs.eight_to_nine = 0.51)
  (h3 : probs.seven_to_eight = 0.15)
  (h4 : probs.six_to_seven = 0.09) :
  (probs.nine_or_above + probs.eight_to_nine = 0.69) ∧
  (probs.nine_or_above + probs.eight_to_nine + probs.seven_to_eight + probs.six_to_seven = 0.93) :=
by sorry

end shooting_training_probabilities_l1550_155000


namespace translation_theorem_l1550_155050

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation2D where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (p : Point2D) (t : Translation2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem (A B : Point2D) (A' : Point2D) :
  A = Point2D.mk 2 2 →
  B = Point2D.mk (-1) 1 →
  A' = Point2D.mk (-2) (-2) →
  let t : Translation2D := { dx := A'.x - A.x, dy := A'.y - A.y }
  applyTranslation B t = Point2D.mk (-5) (-3) := by
  sorry

end translation_theorem_l1550_155050


namespace correct_calculation_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_l1550_155040

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Statement for the correct option (A)
theorem correct_calculation : -cubeRoot 8 = -2 := by sorry

-- Statements for the incorrect options (B, C, D)
theorem incorrect_calculation_B : -abs (-3) ≠ 3 := by sorry

theorem incorrect_calculation_C : Real.sqrt 16 ≠ 4 ∧ Real.sqrt 16 ≠ -4 := by sorry

theorem incorrect_calculation_D : -(2^2) ≠ 4 := by sorry

end correct_calculation_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_l1550_155040


namespace shop_annual_rent_per_square_foot_l1550_155089

/-- Calculates the annual rent per square foot of a shop -/
theorem shop_annual_rent_per_square_foot
  (length : ℝ)
  (width : ℝ)
  (monthly_rent : ℝ)
  (h1 : length = 10)
  (h2 : width = 8)
  (h3 : monthly_rent = 2400) :
  (monthly_rent * 12) / (length * width) = 360 := by
  sorry

end shop_annual_rent_per_square_foot_l1550_155089


namespace intersection_points_form_line_l1550_155084

theorem intersection_points_form_line : 
  ∀ (s : ℝ), 
  ∃ (x y : ℝ), 
  (x + 3 * y = 10 * s + 4) ∧ 
  (2 * x - y = 3 * s - 5) → 
  y = (119 / 133) * x + (435 / 133) := by
  sorry

end intersection_points_form_line_l1550_155084


namespace total_assignments_for_20_points_l1550_155098

def homework_assignments (n : ℕ) : ℕ :=
  if n ≤ 4 then n
  else if n ≤ 8 then 4 + 2 * (n - 4)
  else if n ≤ 12 then 12 + 3 * (n - 8)
  else if n ≤ 16 then 24 + 4 * (n - 12)
  else 40 + 5 * (n - 16)

theorem total_assignments_for_20_points :
  homework_assignments 20 = 60 :=
by sorry

end total_assignments_for_20_points_l1550_155098


namespace problem_solution_l1550_155011

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n - 2 else 3 * n

theorem problem_solution (m : ℤ) (h1 : m % 2 = 0) (h2 : g (g (g m)) = 54) : m = 60 := by
  sorry

end problem_solution_l1550_155011


namespace product_of_five_consecutive_integers_divisible_by_120_l1550_155088

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) :
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end product_of_five_consecutive_integers_divisible_by_120_l1550_155088


namespace truck_tunnel_time_l1550_155001

theorem truck_tunnel_time (truck_length : ℝ) (tunnel_length : ℝ) (speed_mph : ℝ) :
  truck_length = 66 →
  tunnel_length = 330 →
  speed_mph = 45 →
  let speed_fps := speed_mph * 5280 / 3600
  let total_distance := tunnel_length + truck_length
  let time := total_distance / speed_fps
  time = 6 := by sorry

end truck_tunnel_time_l1550_155001


namespace total_donation_l1550_155080

def charity_donation (cassandra james stephanie alex : ℕ) : Prop :=
  cassandra = 5000 ∧
  james = cassandra - 276 ∧
  stephanie = 2 * james ∧
  alex = (3 * (cassandra + stephanie)) / 4 ∧
  cassandra + james + stephanie + alex = 31008

theorem total_donation :
  ∃ (cassandra james stephanie alex : ℕ),
    charity_donation cassandra james stephanie alex :=
by
  sorry

end total_donation_l1550_155080


namespace product_no_x_squared_term_l1550_155057

theorem product_no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 2*a*x + a^2) = x^3 + (a^2 - 2*a)*x + a^2) → a = 1/2 := by
  sorry

end product_no_x_squared_term_l1550_155057


namespace derivative_cosh_l1550_155095

open Real

theorem derivative_cosh (x : ℝ) : 
  deriv (fun x => (1/2) * (exp x + exp (-x))) x = (1/2) * (exp x - exp (-x)) := by
  sorry

end derivative_cosh_l1550_155095


namespace existence_of_positive_rationals_l1550_155044

theorem existence_of_positive_rationals (n : ℕ) (h : n ≥ 4) :
  ∃ (k : ℕ) (a : ℕ → ℚ),
    k ≥ 2 ∧
    (∀ i, i ∈ Finset.range k → a i > 0) ∧
    (Finset.sum (Finset.range k) a = n) ∧
    (Finset.prod (Finset.range k) a = n) := by
  sorry

end existence_of_positive_rationals_l1550_155044


namespace pants_cost_l1550_155047

theorem pants_cost (initial_money : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 109)
  (h2 : shirt_cost = 11)
  (h3 : num_shirts = 2)
  (h4 : money_left = 74) :
  initial_money - (num_shirts * shirt_cost) - money_left = 13 := by
  sorry

end pants_cost_l1550_155047


namespace bee_multiple_l1550_155070

theorem bee_multiple (bees_day1 bees_day2 : ℕ) : 
  bees_day1 = 144 → bees_day2 = 432 → bees_day2 / bees_day1 = 3 := by
  sorry

end bee_multiple_l1550_155070


namespace shaded_area_square_with_quarter_circles_l1550_155082

/-- The area of the shaded region in a square with side length 20 cm and four quarter circles
    with radius 10 cm drawn at the corners is 400 - 100π cm². -/
theorem shaded_area_square_with_quarter_circles :
  let square_side : ℝ := 20
  let circle_radius : ℝ := 10
  let square_area : ℝ := square_side ^ 2
  let quarter_circle_area : ℝ := π * circle_radius ^ 2 / 4
  let total_quarter_circles_area : ℝ := 4 * quarter_circle_area
  let shaded_area : ℝ := square_area - total_quarter_circles_area
  shaded_area = 400 - 100 * π := by
  sorry

end shaded_area_square_with_quarter_circles_l1550_155082


namespace smallest_shadow_area_l1550_155072

/-- The smallest area of the shadow cast by a cube onto a plane -/
theorem smallest_shadow_area (a b : ℝ) (h : b > a) (h_pos : a > 0) :
  ∃ (shadow_area : ℝ), shadow_area = (a^2 * b^2) / (b - a)^2 ∧
  ∀ (other_area : ℝ), other_area ≥ shadow_area := by
  sorry

end smallest_shadow_area_l1550_155072


namespace circle_radius_is_one_l1550_155083

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the circle in rectangular coordinates
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem circle_radius_is_one :
  ∀ ρ θ x y : ℝ,
  polar_equation ρ θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  circle_equation x y →
  1 = 1 := by sorry

end circle_radius_is_one_l1550_155083


namespace fraction_equality_l1550_155068

theorem fraction_equality (m n : ℝ) (h : 1/m + 1/n = 7) : 14*m*n/(m+n) = 2 := by
  sorry

end fraction_equality_l1550_155068


namespace power_equation_solution_l1550_155018

theorem power_equation_solution (x y : ℕ+) (h : 2^(x.val + 1) * 4^y.val = 128) : 
  x.val + 2 * y.val = 6 := by
sorry

end power_equation_solution_l1550_155018


namespace smallest_solution_quadratic_equation_l1550_155046

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 3 * (10 * x^2 + 10 * x + 15) - x * (10 * x - 55)
  ∃ x : ℝ, f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = -29/8 :=
by sorry

end smallest_solution_quadratic_equation_l1550_155046


namespace moving_circle_trajectory_l1550_155085

-- Define the circles M₁ and M₂
def M₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def M₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the moving circle M
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency conditions
def externally_tangent (M : MovingCircle) : Prop :=
  M₁ (M.center.1 + M.radius) M.center.2

def internally_tangent (M : MovingCircle) : Prop :=
  M₂ (M.center.1 - M.radius) M.center.2

-- Define the trajectory of the center of M
def trajectory (x y : ℝ) : Prop :=
  x^2/4 + y^2/3 = 1 ∧ x ≠ -2

-- Theorem statement
theorem moving_circle_trajectory (M : MovingCircle) :
  externally_tangent M → internally_tangent M →
  trajectory M.center.1 M.center.2 :=
sorry

end moving_circle_trajectory_l1550_155085


namespace dog_park_theorem_l1550_155028

/-- The total number of dogs barking after a new group joins -/
def total_dogs (initial : ℕ) (multiplier : ℕ) : ℕ :=
  initial + multiplier * initial

/-- Theorem: Given 30 initial dogs and a new group triple the size, the total is 120 dogs -/
theorem dog_park_theorem :
  total_dogs 30 3 = 120 := by
  sorry

end dog_park_theorem_l1550_155028


namespace smallest_sum_of_two_digit_numbers_l1550_155059

def NumberSet : Finset Nat := {5, 6, 7, 8, 9}

def is_valid_pair (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ a ≠ b ∧ a ≥ 10 ∧ a < 100 ∧ b ≥ 10 ∧ b < 100

def sum_of_pair (a b : Nat) : Nat := a + b

theorem smallest_sum_of_two_digit_numbers :
  ∃ (a b : Nat), is_valid_pair a b ∧
    sum_of_pair a b = 125 ∧
    (∀ (c d : Nat), is_valid_pair c d → sum_of_pair c d ≥ 125) :=
sorry

end smallest_sum_of_two_digit_numbers_l1550_155059


namespace athlete_calorie_burn_l1550_155081

/-- Calculates the total calories burned by an athlete during exercise -/
theorem athlete_calorie_burn 
  (running_rate : ℕ) 
  (walking_rate : ℕ) 
  (total_time : ℕ) 
  (running_time : ℕ) 
  (h1 : running_rate = 10)
  (h2 : walking_rate = 4)
  (h3 : total_time = 60)
  (h4 : running_time = 35)
  (h5 : running_time ≤ total_time) :
  running_rate * running_time + walking_rate * (total_time - running_time) = 450 := by
  sorry

#check athlete_calorie_burn

end athlete_calorie_burn_l1550_155081


namespace probability_at_most_one_incorrect_l1550_155042

/-- The probability of at most one incorrect result in 10 hemoglobin tests -/
def prob_at_most_one_incorrect (p : ℝ) : ℝ :=
  p^9 * (10 - 9*p)

/-- Theorem: Given the accuracy of a hemoglobin test is p, 
    the probability of at most one incorrect result out of 10 tests 
    is equal to p^9 * (10 - 9p) -/
theorem probability_at_most_one_incorrect 
  (p : ℝ) 
  (h1 : 0 ≤ p) 
  (h2 : p ≤ 1) : 
  (p^10 + 10 * (1 - p) * p^9) = prob_at_most_one_incorrect p :=
sorry

end probability_at_most_one_incorrect_l1550_155042


namespace pen_count_l1550_155029

/-- The number of pens in Maria's desk drawer -/
theorem pen_count (red : ℕ) (black : ℕ) (blue : ℕ) (green : ℕ) : 
  red = 8 →
  black = 2 * red →
  blue = black + 5 →
  green = blue / 2 →
  red + black + blue + green = 55 := by
sorry

end pen_count_l1550_155029


namespace max_blanks_proof_l1550_155064

/-- The width of the plywood sheet -/
def plywood_width : ℕ := 22

/-- The height of the plywood sheet -/
def plywood_height : ℕ := 15

/-- The width of the rectangular blank -/
def blank_width : ℕ := 3

/-- The height of the rectangular blank -/
def blank_height : ℕ := 5

/-- The maximum number of rectangular blanks that can be cut from the plywood sheet -/
def max_blanks : ℕ := 22

theorem max_blanks_proof :
  (plywood_width * plywood_height) ≥ (max_blanks * blank_width * blank_height) ∧
  (plywood_width * plywood_height) < ((max_blanks + 1) * blank_width * blank_height) :=
by sorry

end max_blanks_proof_l1550_155064


namespace parallelogram_area_l1550_155041

def v : Fin 2 → ℝ
| 0 => 7
| 1 => -5

def w : Fin 2 → ℝ
| 0 => 14
| 1 => -4

theorem parallelogram_area : 
  abs (v 0 * w 1 - v 1 * w 0) = 42 := by
  sorry

end parallelogram_area_l1550_155041


namespace intersection_of_M_and_N_l1550_155014

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {y | ∃ x ∈ M, y = 2 * x}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := by
  sorry

end intersection_of_M_and_N_l1550_155014


namespace greatest_divisor_XYXY_l1550_155033

/-- A four-digit palindrome of the pattern XYXY -/
def XYXY (X Y : Nat) : Nat := 1000 * X + 100 * Y + 10 * X + Y

/-- Predicate to check if a number is a single digit -/
def is_single_digit (n : Nat) : Prop := n ≥ 0 ∧ n ≤ 9

/-- The theorem stating that 11 is the greatest divisor of all XYXY palindromes -/
theorem greatest_divisor_XYXY :
  ∀ X Y : Nat, is_single_digit X → is_single_digit Y →
  (∀ d : Nat, d > 11 → ¬(d ∣ XYXY X Y)) ∧
  (11 ∣ XYXY X Y) :=
sorry

end greatest_divisor_XYXY_l1550_155033


namespace same_color_probability_l1550_155008

/-- The number of green balls in the bag -/
def green_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 6

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls : ℕ := green_balls + red_balls + blue_balls

/-- The probability of drawing two balls of the same color with replacement -/
theorem same_color_probability : 
  (green_balls^2 + red_balls^2 + blue_balls^2) / total_balls^2 = 101 / 225 := by
  sorry

end same_color_probability_l1550_155008


namespace exactly_one_true_l1550_155002

def X : Set Int := {x | -2 < x ∧ x ≤ 3}

def p (a : ℝ) : Prop := ∀ x ∈ X, (1/3 : ℝ) * x^2 < 2*a - 3

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a = 0

theorem exactly_one_true (a : ℝ) : 
  (p a ∧ ¬(q a)) ∨ (¬(p a) ∧ q a) ↔ a ≤ 1 ∨ a > 3 :=
sorry

end exactly_one_true_l1550_155002


namespace solve_temperature_l1550_155015

def temperature_problem (temps : List ℝ) (avg : ℝ) : Prop :=
  temps.length = 6 ∧
  (temps.sum + (7 * avg - temps.sum)) / 7 = avg

theorem solve_temperature (temps : List ℝ) (avg : ℝ) 
  (h : temperature_problem temps avg) : ℝ :=
  7 * avg - temps.sum

#check solve_temperature

end solve_temperature_l1550_155015


namespace victor_deck_count_l1550_155043

theorem victor_deck_count (cost_per_deck : ℕ) (friend_deck_count : ℕ) (total_spent : ℕ) : ℕ :=
  let victor_deck_count := (total_spent - friend_deck_count * cost_per_deck) / cost_per_deck
  have h1 : cost_per_deck = 8 := by sorry
  have h2 : friend_deck_count = 2 := by sorry
  have h3 : total_spent = 64 := by sorry
  have h4 : victor_deck_count = 6 := by sorry
  victor_deck_count

#check victor_deck_count

end victor_deck_count_l1550_155043


namespace davantes_boy_friends_l1550_155069

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define Davante's total number of friends
def total_friends : ℕ := 2 * days_in_week

-- Define the number of Davante's friends who are girls
def girl_friends : ℕ := 3

-- Theorem statement
theorem davantes_boy_friends :
  total_friends - girl_friends = 11 :=
sorry

end davantes_boy_friends_l1550_155069
