import Mathlib

namespace power_diff_reciprocal_power_l1747_174731

theorem power_diff_reciprocal_power (x : ℂ) :
  x - (1 / x) = Complex.I * Real.sqrt 2 →
  x^2187 - (1 / x^2187) = Complex.I * Real.sqrt 2 := by
  sorry

end power_diff_reciprocal_power_l1747_174731


namespace total_bad_produce_l1747_174795

-- Define the number of carrots and tomatoes picked by each person
def vanessa_carrots : ℕ := 17
def vanessa_tomatoes : ℕ := 12
def mom_carrots : ℕ := 14
def mom_tomatoes : ℕ := 22
def brother_carrots : ℕ := 6
def brother_tomatoes : ℕ := 8

-- Define the number of good carrots and tomatoes
def good_carrots : ℕ := 28
def good_tomatoes : ℕ := 35

-- Define the total number of carrots and tomatoes picked
def total_carrots : ℕ := vanessa_carrots + mom_carrots + brother_carrots
def total_tomatoes : ℕ := vanessa_tomatoes + mom_tomatoes + brother_tomatoes

-- Define the number of bad carrots and tomatoes
def bad_carrots : ℕ := total_carrots - good_carrots
def bad_tomatoes : ℕ := total_tomatoes - good_tomatoes

-- Theorem to prove
theorem total_bad_produce : bad_carrots + bad_tomatoes = 16 := by
  sorry

end total_bad_produce_l1747_174795


namespace truck_speed_difference_l1747_174704

/-- Represents the speed difference between paved and dirt roads for a semi truck journey --/
theorem truck_speed_difference 
  (total_distance : ℝ) 
  (paved_time dirt_time : ℝ) 
  (dirt_speed : ℝ) :
  total_distance = 200 →
  paved_time = 2 →
  dirt_time = 3 →
  dirt_speed = 32 →
  (total_distance - dirt_speed * dirt_time) / paved_time - dirt_speed = 20 := by
  sorry

#check truck_speed_difference

end truck_speed_difference_l1747_174704


namespace simplify_product_of_roots_l1747_174767

theorem simplify_product_of_roots : 
  Real.sqrt (5 * 3) * Real.sqrt (3^4 * 5^2) = 15 * Real.sqrt 15 := by
  sorry

end simplify_product_of_roots_l1747_174767


namespace square_root_of_square_l1747_174747

theorem square_root_of_square (x : ℝ) : {y : ℝ | y^2 = x^2} = {x, -x} := by sorry

end square_root_of_square_l1747_174747


namespace blue_chairs_count_l1747_174706

/-- Represents the number of chairs of each color in a classroom --/
structure Classroom where
  blue : ℕ
  green : ℕ
  white : ℕ

/-- Defines the conditions for the classroom chair problem --/
def validClassroom (c : Classroom) : Prop :=
  c.green = 3 * c.blue ∧
  c.white = c.blue + c.green - 13 ∧
  c.blue + c.green + c.white = 67

/-- Theorem stating that in a valid classroom, there are 10 blue chairs --/
theorem blue_chairs_count (c : Classroom) (h : validClassroom c) : c.blue = 10 := by
  sorry

end blue_chairs_count_l1747_174706


namespace circle_area_ratio_l1747_174736

theorem circle_area_ratio (r R : ℝ) (h : r = R / 3) :
  (π * r^2) / (π * R^2) = 1 / 9 := by
sorry

end circle_area_ratio_l1747_174736


namespace roots_sum_of_squares_l1747_174785

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 9*a + 9 = 0) → (b^2 - 9*b + 9 = 0) → a^2 + b^2 = 63 := by
  sorry

end roots_sum_of_squares_l1747_174785


namespace chapters_undetermined_l1747_174775

/-- Represents a book with a number of pages and chapters -/
structure Book where
  pages : ℕ
  chapters : ℕ

/-- Represents Jake's reading progress -/
structure ReadingProgress where
  initialRead : ℕ
  laterRead : ℕ
  totalRead : ℕ

/-- Given the conditions of Jake's reading and the book, 
    prove that the number of chapters cannot be determined -/
theorem chapters_undetermined (book : Book) (progress : ReadingProgress) : 
  book.pages = 95 ∧ 
  progress.initialRead = 37 ∧ 
  progress.laterRead = 25 ∧ 
  progress.totalRead = 62 →
  ¬ ∃ (n : ℕ), ∀ (b : Book), 
    b.pages = book.pages ∧ 
    b.chapters = n :=
by sorry

end chapters_undetermined_l1747_174775


namespace inequality_holds_l1747_174763

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^3 + k * log x

theorem inequality_holds (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : k ≥ -3) 
  (h2 : x₁ ≥ 1) 
  (h3 : x₂ ≥ 1) 
  (h4 : x₁ > x₂) : 
  (deriv (f k) x₁ + deriv (f k) x₂) / 2 > (f k x₁ - f k x₂) / (x₁ - x₂) :=
by sorry

end inequality_holds_l1747_174763


namespace sin_theta_value_l1747_174717

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 2 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = (-5 + Real.sqrt 29) / 2 := by
  sorry

end sin_theta_value_l1747_174717


namespace phi_value_l1747_174751

open Real

theorem phi_value (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = sin x * cos φ + cos x * sin φ) →
  (0 < φ) →
  (φ < π) →
  (f (2 * (π/6) + π/6) = 1/2) →
  φ = π/3 := by
sorry

end phi_value_l1747_174751


namespace nonzero_terms_count_l1747_174739

/-- The number of nonzero terms in the expansion of (x+4)(3x^3 + 2x^2 + 3x + 9) - 4(x^4 - 3x^3 + 2x^2 + 7x) -/
theorem nonzero_terms_count (x : ℝ) : 
  let expansion := (x + 4) * (3*x^3 + 2*x^2 + 3*x + 9) - 4*(x^4 - 3*x^3 + 2*x^2 + 7*x)
  ∃ (a b c d e : ℝ), 
    expansion = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 :=
by sorry

end nonzero_terms_count_l1747_174739


namespace angle_relationship_l1747_174735

theorem angle_relationship (α β : Real) 
  (h1 : 0 < α) 
  (h2 : α < 2 * β) 
  (h3 : 2 * β ≤ π / 2) 
  (h4 : 2 * Real.cos (α + β) * Real.cos β = -1 + 2 * Real.sin (α + β) * Real.sin β) : 
  α + 2 * β = 2 * π / 3 := by
  sorry

end angle_relationship_l1747_174735


namespace inverse_of_A_zero_matrix_if_not_invertible_l1747_174782

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 7; 2, 6]

theorem inverse_of_A :
  let inv_A := !![0.6, -0.7; -0.2, 0.4]
  A.det ≠ 0 → A * inv_A = 1 ∧ inv_A * A = 1 :=
by sorry

theorem zero_matrix_if_not_invertible :
  A.det = 0 → A⁻¹ = 0 :=
by sorry

end inverse_of_A_zero_matrix_if_not_invertible_l1747_174782


namespace units_digit_of_3_pow_4_l1747_174703

theorem units_digit_of_3_pow_4 : (3^4 : ℕ) % 10 = 1 := by sorry

end units_digit_of_3_pow_4_l1747_174703


namespace new_person_weight_l1747_174743

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 70 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 90 :=
by sorry

end new_person_weight_l1747_174743


namespace james_dance_duration_l1747_174707

/-- Represents the number of calories burned per hour while walking -/
def calories_walking : ℕ := 300

/-- Represents the number of calories burned per week from dancing -/
def calories_dancing_weekly : ℕ := 2400

/-- Represents the number of times James dances per week -/
def dance_sessions_per_week : ℕ := 4

/-- Represents the ratio of calories burned dancing compared to walking -/
def dancing_to_walking_ratio : ℕ := 2

/-- Proves that James dances for 1 hour each time given the conditions -/
theorem james_dance_duration :
  (calories_dancing_weekly / (dancing_to_walking_ratio * calories_walking)) / dance_sessions_per_week = 1 :=
by sorry

end james_dance_duration_l1747_174707


namespace log_three_seven_l1747_174734

theorem log_three_seven (a b : ℝ) (h1 : Real.log 2 / Real.log 3 = a) (h2 : Real.log 7 / Real.log 2 = b) :
  Real.log 7 / Real.log 3 = a * b := by
  sorry

end log_three_seven_l1747_174734


namespace maxim_birth_probability_l1747_174744

/-- The year Maxim starts school -/
def school_start_year : ℕ := 2014

/-- The month Maxim starts school (1-based) -/
def school_start_month : ℕ := 9

/-- The day Maxim starts school -/
def school_start_day : ℕ := 1

/-- Maxim's age when starting school -/
def school_start_age : ℕ := 6

/-- Whether the school start date is Maxim's birthday -/
def is_birthday : Prop := False

/-- The number of days from Jan 1, 2008 to Aug 31, 2008 inclusive -/
def days_in_2008 : ℕ := 244

/-- The total number of possible birth dates -/
def total_possible_days : ℕ := 365

/-- The probability that Maxim was born in 2008 -/
def prob_born_2008 : ℚ := days_in_2008 / total_possible_days

theorem maxim_birth_probability : 
  prob_born_2008 = 244 / 365 := by sorry

end maxim_birth_probability_l1747_174744


namespace r_fourth_plus_inverse_r_fourth_l1747_174756

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 := by
  sorry

end r_fourth_plus_inverse_r_fourth_l1747_174756


namespace olivia_money_distribution_l1747_174732

/-- Prove that Olivia needs to give 2 euros to each sister for equal distribution -/
theorem olivia_money_distribution (olivia_initial : ℕ) (sister_initial : ℕ) (num_sisters : ℕ) 
  (olivia_gives : ℕ) :
  olivia_initial = 20 →
  sister_initial = 10 →
  num_sisters = 4 →
  olivia_gives = 2 →
  (olivia_initial - num_sisters * olivia_gives = 
   sister_initial + olivia_gives) ∧
  (olivia_initial + num_sisters * sister_initial = 
   (num_sisters + 1) * (sister_initial + olivia_gives)) :=
by sorry

end olivia_money_distribution_l1747_174732


namespace part_one_part_two_l1747_174797

-- Define the new operation ※
def star (a b : ℝ) : ℝ := a^2 - b^2

-- Theorem for part 1
theorem part_one : star 2 (-4) = -12 := by sorry

-- Theorem for part 2
theorem part_two : ∀ x : ℝ, star (x + 5) 3 = 0 ↔ x = -8 ∨ x = -2 := by sorry

end part_one_part_two_l1747_174797


namespace twelfth_term_of_specific_sequence_l1747_174758

/-- Given a geometric sequence with first term a₁ and common ratio r,
    the nth term is given by aₙ = a₁ * r^(n-1) -/
def geometric_sequence (a₁ : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  a₁ * r^(n-1)

/-- The 12th term of a geometric sequence with first term 5 and common ratio -3 is -885735 -/
theorem twelfth_term_of_specific_sequence :
  geometric_sequence 5 (-3) 12 = -885735 := by
  sorry

end twelfth_term_of_specific_sequence_l1747_174758


namespace fermat_little_theorem_general_l1747_174742

theorem fermat_little_theorem_general (p : ℕ) (m : ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, m^p - m = k * p :=
sorry

end fermat_little_theorem_general_l1747_174742


namespace residue_14_power_2046_mod_17_l1747_174789

theorem residue_14_power_2046_mod_17 : 14^2046 % 17 = 12 := by
  sorry

end residue_14_power_2046_mod_17_l1747_174789


namespace interest_calculation_l1747_174771

/-- Calculates the simple interest and final amount given initial principal, annual rate, and time in years -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ × ℝ :=
  let interest := principal * rate * time
  let final_amount := principal + interest
  (interest, final_amount)

theorem interest_calculation (P : ℝ) :
  let (interest, final_amount) := simple_interest P 0.06 0.25
  final_amount = 510.60 → interest = 7.54 := by
sorry

end interest_calculation_l1747_174771


namespace baseball_division_games_l1747_174719

theorem baseball_division_games 
  (N M : ℕ) 
  (h1 : N > 2 * M) 
  (h2 : M > 4) 
  (h3 : 2 * N + 5 * M = 82) : 
  2 * N = 52 := by
  sorry

end baseball_division_games_l1747_174719


namespace cards_ratio_l1747_174784

/-- Prove the ratio of cards given to initial cards is 1:2 -/
theorem cards_ratio (brandon_cards : ℕ) (malcom_extra : ℕ) (malcom_left : ℕ)
  (h1 : brandon_cards = 20)
  (h2 : malcom_extra = 8)
  (h3 : malcom_left = 14) :
  (brandon_cards + malcom_extra - malcom_left) / (brandon_cards + malcom_extra) = 1 / 2 :=
by sorry

end cards_ratio_l1747_174784


namespace sum_at_two_and_neg_two_l1747_174708

/-- A cubic polynomial Q with specific properties -/
structure CubicPolynomial (p : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ (a b c : ℝ), ∀ x, Q x = a * x^3 + b * x^2 + c * x + p
  at_zero : Q 0 = p
  at_one : Q 1 = 3 * p
  at_neg_one : Q (-1) = 4 * p

/-- The sum of Q(2) and Q(-2) for a specific cubic polynomial Q -/
theorem sum_at_two_and_neg_two (p : ℝ) (Q : CubicPolynomial p) :
  Q.Q 2 + Q.Q (-2) = 22 * p := by
  sorry

end sum_at_two_and_neg_two_l1747_174708


namespace smaller_bill_value_l1747_174755

/-- The value of the smaller denomination bill -/
def x : ℕ := sorry

/-- The total number of bills Anna has -/
def total_bills : ℕ := 12

/-- The number of smaller denomination bills Anna has -/
def smaller_bills : ℕ := 4

/-- The value of a $10 bill -/
def ten_dollar : ℕ := 10

/-- The total value of all bills in dollars -/
def total_value : ℕ := 100

theorem smaller_bill_value :
  x * smaller_bills + (total_bills - smaller_bills) * ten_dollar = total_value ∧ x = 5 := by
  sorry

end smaller_bill_value_l1747_174755


namespace angle_C_measure_l1747_174722

theorem angle_C_measure (A B C : ℝ) (h : A + B = 80) : A + B + C = 180 → C = 100 := by
  sorry

end angle_C_measure_l1747_174722


namespace quinary_444_equals_octal_174_l1747_174774

/-- Converts a quinary (base 5) number to decimal (base 10) -/
def quinary_to_decimal (q : ℕ) : ℕ :=
  (q / 100) * 5^2 + ((q / 10) % 10) * 5^1 + (q % 10) * 5^0

/-- Converts a decimal (base 10) number to octal (base 8) -/
def decimal_to_octal (d : ℕ) : ℕ :=
  (d / 64) * 100 + ((d / 8) % 8) * 10 + (d % 8)

/-- Theorem stating that 444₅ is equal to 174₈ -/
theorem quinary_444_equals_octal_174 :
  decimal_to_octal (quinary_to_decimal 444) = 174 := by
  sorry

end quinary_444_equals_octal_174_l1747_174774


namespace intersection_distance_proof_l1747_174702

/-- The distance between intersection points of y = 5 and y = 3x^2 + 2x - 2 -/
def intersection_distance : ℝ := sorry

/-- The equation y = 3x^2 + 2x - 2 -/
def parabola (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

theorem intersection_distance_proof :
  let p : ℕ := 88
  let q : ℕ := 3
  (∃ (x₁ x₂ : ℝ), 
    parabola x₁ = 5 ∧ 
    parabola x₂ = 5 ∧ 
    x₁ ≠ x₂ ∧
    intersection_distance = |x₁ - x₂|) ∧
  intersection_distance = Real.sqrt p / q ∧
  p - q^2 = 79 := by sorry

end intersection_distance_proof_l1747_174702


namespace puppies_per_cage_l1747_174788

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 78) 
  (h2 : sold_puppies = 30) 
  (h3 : num_cages = 6) 
  (h4 : initial_puppies > sold_puppies) :
  (initial_puppies - sold_puppies) / num_cages = 8 := by
  sorry

end puppies_per_cage_l1747_174788


namespace inequality_proof_l1747_174757

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end inequality_proof_l1747_174757


namespace complex_number_quadrant_l1747_174776

theorem complex_number_quadrant : ∃ (z : ℂ), z = 2 / (1 + Complex.I) ∧ Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end complex_number_quadrant_l1747_174776


namespace qin_jiushao_v3_value_main_theorem_l1747_174705

def f (x : ℝ) : ℝ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def qin_jiushao_v3 (a b c d : ℝ) (x : ℝ) : ℝ :=
  ((a * x + b) * x + c) * x + d

theorem qin_jiushao_v3_value :
  qin_jiushao_v3 2 5 6 23 (-4) = -49 :=
by sorry

-- The main theorem
theorem main_theorem :
  ∃ (v3 : ℝ), qin_jiushao_v3 2 5 6 23 (-4) = v3 ∧ v3 = -49 :=
by sorry

end qin_jiushao_v3_value_main_theorem_l1747_174705


namespace line_ellipse_intersection_l1747_174730

/-- Given a line y = x + m intersecting the ellipse 4x^2 + y^2 = 1 and forming a chord of length 2√2/5, prove that m = ± √5/2 -/
theorem line_ellipse_intersection (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    4 * x₁^2 + (x₁ + m)^2 = 1 ∧ 
    4 * x₂^2 + (x₂ + m)^2 = 1 ∧ 
    (x₂ - x₁)^2 + ((x₂ + m) - (x₁ + m))^2 = (2 * Real.sqrt 2 / 5)^2) → 
  m = Real.sqrt 5 / 2 ∨ m = -Real.sqrt 5 / 2 :=
by sorry


end line_ellipse_intersection_l1747_174730


namespace pine_saplings_in_sample_l1747_174787

theorem pine_saplings_in_sample 
  (total_saplings : ℕ) 
  (pine_saplings : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_saplings = 20000) 
  (h2 : pine_saplings = 4000) 
  (h3 : sample_size = 100) : 
  (sample_size * pine_saplings) / total_saplings = 20 := by
sorry

end pine_saplings_in_sample_l1747_174787


namespace double_root_k_l1747_174780

/-- A cubic equation with a double root -/
def has_double_root (k : ℝ) : Prop :=
  ∃ (r s : ℝ), (∀ x, x^3 + k*x - 128 = (x - r)^2 * (x - s))

/-- The value of k for which x^3 + kx - 128 = 0 has a double root -/
theorem double_root_k : ∃ k : ℝ, has_double_root k ∧ k = -48 := by
  sorry

end double_root_k_l1747_174780


namespace gcd_8251_6105_l1747_174766

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l1747_174766


namespace part_one_part_two_l1747_174713

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f a x < 4 - 2*a ↔ -4 < x ∧ x < 4) → a = 0 := 
sorry

-- Part 2
theorem part_two : 
  (∀ m : ℝ, (∀ x : ℝ, f 1 x - f 1 (-2*x) ≤ x + m) ↔ 2 ≤ m) :=
sorry

end part_one_part_two_l1747_174713


namespace sufficient_not_necessary_condition_l1747_174762

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → x / y > 1) ∧
  ∃ x y : ℝ, x / y > 1 ∧ ¬(x > y ∧ y > 0) :=
by sorry

end sufficient_not_necessary_condition_l1747_174762


namespace unique_number_l1747_174754

def is_valid_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ),
    n = 100 * x + 10 * y + z ∧
    x ≥ 1 ∧ x ≤ 9 ∧
    y ≥ 0 ∧ y ≤ 9 ∧
    z ≥ 0 ∧ z ≤ 9 ∧
    100 * z + 10 * y + x = n + 198 ∧
    100 * x + 10 * z + y = n + 9 ∧
    x^2 + y^2 + z^2 = 4 * (x + y + z) + 2

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 345 :=
sorry

end unique_number_l1747_174754


namespace weight_removed_l1747_174714

/-- Given weights of sugar and salt bags, and their combined weight after removal,
    prove the amount of weight removed. -/
theorem weight_removed (sugar_weight salt_weight new_combined_weight : ℕ)
  (h1 : sugar_weight = 16)
  (h2 : salt_weight = 30)
  (h3 : new_combined_weight = 42) :
  sugar_weight + salt_weight - new_combined_weight = 4 := by
  sorry

end weight_removed_l1747_174714


namespace missing_number_proof_l1747_174712

theorem missing_number_proof : ∃ x : ℚ, (306 / 34) * 15 + x = 405 := by
  sorry

end missing_number_proof_l1747_174712


namespace floor_sqrt_120_l1747_174729

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end floor_sqrt_120_l1747_174729


namespace total_units_is_531_l1747_174778

/-- A mixed-use development with various floor types and unit distributions -/
structure MixedUseDevelopment where
  total_floors : Nat
  regular_floors : Nat
  luxury_floors : Nat
  penthouse_floors : Nat
  commercial_floors : Nat
  other_floors : Nat
  regular_odd_units : Nat
  regular_even_units : Nat
  luxury_avg_units : Nat
  penthouse_units : Nat
  commercial_units : Nat
  amenities_uncounted_units : Nat
  other_uncounted_units : Nat

/-- Calculate the total number of units in the mixed-use development -/
def total_units (dev : MixedUseDevelopment) : Nat :=
  let regular_units := (dev.regular_floors / 2 + dev.regular_floors % 2) * dev.regular_odd_units +
                       (dev.regular_floors / 2) * dev.regular_even_units
  let luxury_units := dev.luxury_floors * dev.luxury_avg_units
  let penthouse_units := dev.penthouse_floors * dev.penthouse_units
  let commercial_units := dev.commercial_floors * dev.commercial_units
  let uncounted_units := dev.amenities_uncounted_units + dev.other_uncounted_units
  regular_units + luxury_units + penthouse_units + commercial_units + uncounted_units

/-- The mixed-use development described in the problem -/
def problem_development : MixedUseDevelopment where
  total_floors := 60
  regular_floors := 25
  luxury_floors := 20
  penthouse_floors := 10
  commercial_floors := 3
  other_floors := 2
  regular_odd_units := 14
  regular_even_units := 12
  luxury_avg_units := 8
  penthouse_units := 2
  commercial_units := 5
  amenities_uncounted_units := 4
  other_uncounted_units := 6

/-- Theorem stating that the total number of units in the problem development is 531 -/
theorem total_units_is_531 : total_units problem_development = 531 := by
  sorry


end total_units_is_531_l1747_174778


namespace carpet_area_calculation_l1747_174746

/-- Calculates the area of carpet required for a room and corridor -/
theorem carpet_area_calculation (main_length main_width corridor_length corridor_width : ℝ) 
  (h_main_length : main_length = 15)
  (h_main_width : main_width = 12)
  (h_corridor_length : corridor_length = 10)
  (h_corridor_width : corridor_width = 3)
  (h_feet_to_yard : 3 = 1) :
  (main_length * main_width + corridor_length * corridor_width) / 9 = 23.33 := by
sorry

#eval (15 * 12 + 10 * 3) / 9

end carpet_area_calculation_l1747_174746


namespace compound_interest_rate_calculation_l1747_174725

theorem compound_interest_rate_calculation
  (P : ℝ)  -- Principal amount
  (CI : ℝ)  -- Compound Interest
  (t : ℝ)  -- Time in years
  (n : ℝ)  -- Number of times interest is compounded per year
  (h1 : P = 8000)
  (h2 : CI = 484.76847061839544)
  (h3 : t = 1.5)
  (h4 : n = 2)
  : ∃ (r : ℝ), abs (r - 0.0397350993377484) < 0.0000000000000001 :=
by
  sorry

end compound_interest_rate_calculation_l1747_174725


namespace subset_range_of_a_l1747_174701

theorem subset_range_of_a (a : ℝ) : 
  let A := {x : ℝ | 1 ≤ x ∧ x ≤ 5}
  let B := {x : ℝ | a < x ∧ x < a + 1}
  B ⊆ A → 1 ≤ a ∧ a ≤ 4 :=
by sorry

end subset_range_of_a_l1747_174701


namespace construction_delay_l1747_174764

/-- Represents the construction project with given parameters -/
structure ConstructionProject where
  initialWorkers : ℕ
  additionalWorkers : ℕ
  daysBeforeAddingWorkers : ℕ
  totalDays : ℕ

/-- Calculates the total man-days for the project -/
def totalManDays (project : ConstructionProject) : ℕ :=
  (project.initialWorkers * project.daysBeforeAddingWorkers) +
  ((project.initialWorkers + project.additionalWorkers) * (project.totalDays - project.daysBeforeAddingWorkers))

/-- Calculates the number of days needed with only initial workers -/
def daysWithInitialWorkersOnly (project : ConstructionProject) : ℕ :=
  (totalManDays project) / project.initialWorkers

/-- Theorem stating the delay in construction without additional workers -/
theorem construction_delay (project : ConstructionProject) 
  (h1 : project.initialWorkers = 100)
  (h2 : project.additionalWorkers = 100)
  (h3 : project.daysBeforeAddingWorkers = 10)
  (h4 : project.totalDays = 100) :
  daysWithInitialWorkersOnly project - project.totalDays = 90 := by
  sorry


end construction_delay_l1747_174764


namespace z_coord_for_specific_line_l1747_174700

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- The z-coordinate of a point on the line when its y-coordinate is given -/
def z_coord_at_y (l : Line3D) (y : ℝ) : ℝ :=
  sorry

theorem z_coord_for_specific_line :
  let l : Line3D := { point1 := (3, 3, 2), point2 := (6, 2, -1) }
  z_coord_at_y l 1 = -4 := by
  sorry

end z_coord_for_specific_line_l1747_174700


namespace marble_problem_l1747_174752

theorem marble_problem (b : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = b)
  (h2 : brian = 3 * b)
  (h3 : caden = 4 * brian)
  (h4 : daryl = 6 * caden)
  (h5 : angela + brian + caden + daryl = 312) :
  b = 39 / 11 := by
sorry

end marble_problem_l1747_174752


namespace martha_cookies_theorem_l1747_174727

/-- Given that Martha can make 24 cookies with 3 cups of flour, this function
    calculates how many cookies she can make with a given number of cups. -/
def cookies_from_flour (cups : ℚ) : ℚ :=
  (24 / 3) * cups

/-- Given that Martha can make 24 cookies with 3 cups of flour, this function
    calculates how many cups of flour are needed to make a given number of cookies. -/
def flour_for_cookies (cookies : ℚ) : ℚ :=
  (3 / 24) * cookies

/-- Theorem stating that with 5 cups of flour, Martha can make 40 cookies,
    and 60 cookies require 7.5 cups of flour. -/
theorem martha_cookies_theorem :
  cookies_from_flour 5 = 40 ∧ flour_for_cookies 60 = 7.5 := by
  sorry

end martha_cookies_theorem_l1747_174727


namespace interest_difference_theorem_l1747_174770

/-- Proves that given an interest rate of 5% per annum for 2 years, 
    if the difference between compound interest and simple interest is 18, 
    then the principal amount is 7200. -/
theorem interest_difference_theorem (P : ℝ) : 
  P * (1 + 0.05)^2 - P - (P * 0.05 * 2) = 18 → P = 7200 := by
  sorry

end interest_difference_theorem_l1747_174770


namespace f_2007_values_l1747_174720

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1

theorem f_2007_values (f : ℕ → ℕ) (h : is_valid_f f) : 
  f 2007 ∈ Finset.range 2009 ∧ 
  ∀ k ∈ Finset.range 2009, ∃ g : ℕ → ℕ, is_valid_f g ∧ g 2007 = k :=
sorry

end f_2007_values_l1747_174720


namespace min_value_quadratic_root_condition_l1747_174724

/-- Given a quadratic equation x^2 + ax + b - 3 = 0 with a real root in [1,2],
    the minimum value of a^2 + (b-4)^2 is 2 -/
theorem min_value_quadratic_root_condition (a b : ℝ) :
  (∃ x : ℝ, x^2 + a*x + b - 3 = 0 ∧ 1 ≤ x ∧ x ≤ 2) →
  (∀ a' b' : ℝ, (∃ x : ℝ, x^2 + a'*x + b' - 3 = 0 ∧ 1 ≤ x ∧ x ≤ 2) →
    a^2 + (b-4)^2 ≤ a'^2 + (b'-4)^2) →
  a^2 + (b-4)^2 = 2 :=
by sorry


end min_value_quadratic_root_condition_l1747_174724


namespace jose_share_of_profit_l1747_174772

/-- Calculates the share of profit for an investor based on their investment, time period, and total profit --/
def calculate_share_of_profit (investment : ℕ) (months : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * months * total_profit) / total_investment_months

theorem jose_share_of_profit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (total_profit : ℕ)
  (h1 : tom_investment = 30000)
  (h2 : tom_months = 12)
  (h3 : jose_investment = 45000)
  (h4 : jose_months = 10)
  (h5 : total_profit = 63000) :
  calculate_share_of_profit jose_investment jose_months (tom_investment * tom_months + jose_investment * jose_months) total_profit = 35000 :=
by sorry

end jose_share_of_profit_l1747_174772


namespace cinnamon_nutmeg_difference_l1747_174740

/-- The amount of cinnamon used in tablespoons -/
def cinnamon : ℚ := 0.6666666666666666

/-- The amount of nutmeg used in tablespoons -/
def nutmeg : ℚ := 0.5

/-- The difference between cinnamon and nutmeg amounts -/
def difference : ℚ := cinnamon - nutmeg

theorem cinnamon_nutmeg_difference : difference = 0.1666666666666666 := by sorry

end cinnamon_nutmeg_difference_l1747_174740


namespace nickel_probability_is_5_24_l1747_174716

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℕ
  | Coin.Dime => 800
  | Coin.Nickel => 500
  | Coin.Penny => 300

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of randomly selecting a nickel from the jar -/
def nickelProbability : ℚ := coinCount Coin.Nickel / totalCoins

theorem nickel_probability_is_5_24 : nickelProbability = 5 / 24 := by
  sorry

end nickel_probability_is_5_24_l1747_174716


namespace simple_interest_problem_l1747_174749

theorem simple_interest_problem (simple_interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  simple_interest = 4016.25 →
  rate = 0.01 →
  time = 3 →
  principal = simple_interest / (rate * time) →
  principal = 133875 := by
sorry

end simple_interest_problem_l1747_174749


namespace particle_motion_l1747_174777

/-- A particle moves under the influence of gravity and an additional constant acceleration. -/
theorem particle_motion
  (V₀ g a t V S : ℝ)
  (hV : V = g * t + a * t + V₀)
  (hS : S = (1/2) * (g + a) * t^2 + V₀ * t) :
  t = (2 * S) / (V + V₀) :=
sorry

end particle_motion_l1747_174777


namespace fraction_simplification_l1747_174728

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 * x^2 - x + 1) / (x^2 - 1) - x / (x - 1) = (x - 1) / (x + 1) := by
  sorry

end fraction_simplification_l1747_174728


namespace total_winter_clothing_l1747_174793

/-- The number of boxes containing winter clothing -/
def num_boxes : ℕ := 3

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 3

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 4

/-- Theorem: The total number of pieces of winter clothing is 21 -/
theorem total_winter_clothing : 
  num_boxes * (scarves_per_box + mittens_per_box) = 21 := by
  sorry

end total_winter_clothing_l1747_174793


namespace gamma_bank_lowest_savings_l1747_174753

def initial_funds : ℝ := 150000
def total_cost : ℝ := 201200

def rebs_bank_interest : ℝ := 2720.33
def gamma_bank_interest : ℝ := 3375.00
def tisi_bank_interest : ℝ := 2349.13
def btv_bank_interest : ℝ := 2264.11

def amount_to_save (interest : ℝ) : ℝ :=
  total_cost - initial_funds - interest

theorem gamma_bank_lowest_savings :
  let rebs_savings := amount_to_save rebs_bank_interest
  let gamma_savings := amount_to_save gamma_bank_interest
  let tisi_savings := amount_to_save tisi_bank_interest
  let btv_savings := amount_to_save btv_bank_interest
  (gamma_savings ≤ rebs_savings) ∧
  (gamma_savings ≤ tisi_savings) ∧
  (gamma_savings ≤ btv_savings) :=
by sorry

end gamma_bank_lowest_savings_l1747_174753


namespace green_beans_count_l1747_174750

theorem green_beans_count (total : ℕ) (red_fraction : ℚ) (white_fraction : ℚ) (green_fraction : ℚ) : 
  total = 572 →
  red_fraction = 1/4 →
  white_fraction = 1/3 →
  green_fraction = 1/2 →
  ∃ (red white green : ℕ),
    red = total * red_fraction ∧
    white = (total - red) * white_fraction ∧
    green = (total - red - white) * green_fraction ∧
    green = 143 :=
by sorry

end green_beans_count_l1747_174750


namespace arithmetic_geometric_ratio_l1747_174768

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_ne_zero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : (a 3) ^ 2 = a 1 * a 9

/-- The main theorem -/
theorem arithmetic_geometric_ratio (seq : ArithmeticSequence) :
  (seq.a 2 + seq.a 4 + seq.a 10) / (seq.a 1 + seq.a 3 + seq.a 9) = 16 / 13 := by
  sorry

end arithmetic_geometric_ratio_l1747_174768


namespace fraction_to_longest_side_is_five_twelfths_l1747_174741

/-- Represents a trapezoid field with corn -/
structure CornField where
  -- Side lengths in clockwise order from a 60° angle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  -- Angles at the non-parallel sides
  angle1 : ℝ
  angle2 : ℝ
  -- Conditions
  side1_eq : side1 = 150
  side2_eq : side2 = 150
  side3_eq : side3 = 200
  side4_eq : side4 = 200
  angle1_eq : angle1 = 60
  angle2_eq : angle2 = 120
  is_trapezoid : angle1 + angle2 = 180

/-- The fraction of the crop brought to the longest side -/
def fractionToLongestSide (field : CornField) : ℚ :=
  5/12

/-- Theorem stating that the fraction of the crop brought to the longest side is 5/12 -/
theorem fraction_to_longest_side_is_five_twelfths (field : CornField) :
  fractionToLongestSide field = 5/12 := by
  sorry

end fraction_to_longest_side_is_five_twelfths_l1747_174741


namespace instantaneous_velocity_at_one_l1747_174773

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 6 * t^2

-- Theorem statement
theorem instantaneous_velocity_at_one :
  velocity 1 = 6 := by sorry

end instantaneous_velocity_at_one_l1747_174773


namespace davids_math_marks_l1747_174718

/-- Represents the marks obtained by David in various subjects -/
structure Marks where
  english : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  mathematics : ℕ

/-- Calculates the average marks given the total marks and number of subjects -/
def average (total : ℕ) (subjects : ℕ) : ℚ :=
  (total : ℚ) / (subjects : ℚ)

/-- Theorem stating that given David's marks in other subjects and his average,
    his Mathematics marks must be 60 -/
theorem davids_math_marks (m : Marks) (h1 : m.english = 72) (h2 : m.physics = 35)
    (h3 : m.chemistry = 62) (h4 : m.biology = 84)
    (h5 : average (m.english + m.physics + m.chemistry + m.biology + m.mathematics) 5 = 62.6) :
    m.mathematics = 60 := by
  sorry

#check davids_math_marks

end davids_math_marks_l1747_174718


namespace solve_equations_l1747_174715

theorem solve_equations :
  (∃ x : ℝ, 4 * x - 3 * (20 - x) + 4 = 0 ∧ x = 8) ∧
  (∃ x : ℝ, (2 * x + 1) / 3 = 1 - (x - 1) / 5 ∧ x = 1) := by
  sorry

end solve_equations_l1747_174715


namespace ellipse_focus_distance_l1747_174723

theorem ellipse_focus_distance (x y : ℝ) :
  x^2 / 25 + y^2 / 16 = 1 →
  ∃ (f1 f2 : ℝ × ℝ), 
    (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧ 
      Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) = 3) →
    Real.sqrt ((x - f2.1)^2 + (y - f2.2)^2) = 7 :=
by sorry

end ellipse_focus_distance_l1747_174723


namespace production_theorem_l1747_174726

/-- Represents the production scenario -/
structure ProductionScenario where
  women : ℕ
  hours_per_day : ℕ
  days : ℕ
  units_produced : ℚ

/-- The production function that calculates the units produced given a scenario -/
def production_function (x : ProductionScenario) (z : ProductionScenario) : ℚ :=
  (z.women * z.hours_per_day * z.days : ℚ) * x.units_produced / (x.women * x.hours_per_day * x.days : ℚ)

theorem production_theorem (x z : ProductionScenario) 
  (h : x.women = x.hours_per_day ∧ x.hours_per_day = x.days ∧ x.units_produced = x.women ^ 2) :
  production_function x z = (z.women * z.hours_per_day * z.days : ℚ) / x.women := by
  sorry

#check production_theorem

end production_theorem_l1747_174726


namespace set_S_properties_l1747_174733

-- Define the set S
def S (m n : ℝ) : Set ℝ := {x : ℝ | m ≤ x ∧ x ≤ n}

-- Main theorem
theorem set_S_properties (m n : ℝ) (h_nonempty : Set.Nonempty (S m n))
    (h_closed : ∀ x ∈ S m n, x^2 ∈ S m n) :
  (m = -1/2 → 1/4 ≤ n ∧ n ≤ 1) ∧
  (n = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0) := by
  sorry

end set_S_properties_l1747_174733


namespace product_of_three_numbers_l1747_174791

theorem product_of_three_numbers (x y z m : ℚ) : 
  x + y + z = 120 → 
  5 * x = m → 
  y - 12 = m → 
  z + 12 = m → 
  x ≤ y ∧ x ≤ z → 
  y ≥ z → 
  x * y * z = 4095360 / 1331 := by
sorry

end product_of_three_numbers_l1747_174791


namespace quadratic_roots_sum_and_product_l1747_174796

/-- Given a quadratic equation x^2 + 4x - 1 = 0 with roots m and n, prove that m + n + mn = -5 -/
theorem quadratic_roots_sum_and_product (m n : ℝ) : 
  (∀ x, x^2 + 4*x - 1 = 0 ↔ x = m ∨ x = n) → m + n + m*n = -5 := by
  sorry

end quadratic_roots_sum_and_product_l1747_174796


namespace pie_distribution_probability_l1747_174786

/-- Represents the total number of pies -/
def total_pies : ℕ := 6

/-- Represents the number of growth pies -/
def growth_pies : ℕ := 2

/-- Represents the number of shrink pies -/
def shrink_pies : ℕ := 4

/-- Represents the number of pies given to Mary -/
def pies_given : ℕ := 3

/-- The probability that one of the girls does not have a single growth pie -/
def prob_no_growth_pie : ℚ := 7/10

theorem pie_distribution_probability :
  prob_no_growth_pie = 1 - (Nat.choose shrink_pies (pies_given - 1) : ℚ) / (Nat.choose total_pies pies_given) :=
by sorry

end pie_distribution_probability_l1747_174786


namespace percentage_problem_l1747_174792

theorem percentage_problem (x : ℝ) : 80 = 16.666666666666668 / 100 * x → x = 480 := by
  sorry

end percentage_problem_l1747_174792


namespace trapezoid_perimeter_is_220_l1747_174794

/-- A trapezoid with the given properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  angle_BCD : ℝ

/-- The perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + 2 * t.BC + t.CD

/-- Theorem stating that the perimeter of the given trapezoid is 220 -/
theorem trapezoid_perimeter_is_220 (t : Trapezoid) 
  (h1 : t.AB = 60)
  (h2 : t.CD = 40)
  (h3 : t.angle_BCD = 120 * π / 180)
  (h4 : t.BC = Real.sqrt (t.CD^2 + (t.AB - t.CD)^2 - 2 * t.CD * (t.AB - t.CD) * Real.cos t.angle_BCD)) :
  perimeter t = 220 := by
  sorry

#check trapezoid_perimeter_is_220

end trapezoid_perimeter_is_220_l1747_174794


namespace arithmetic_sequence_sum_l1747_174709

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 8 = 6) →
  (a 3 * a 8 = 5) →
  a 5 + a 6 = 6 := by
sorry

end arithmetic_sequence_sum_l1747_174709


namespace cubic_decreasing_iff_l1747_174745

theorem cubic_decreasing_iff (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => a * x^3 - x) ((3 * a * x^2) - 1) x) →
  (∀ x y : ℝ, x < y → (a * x^3 - x) > (a * y^3 - y)) ↔ a ≤ 0 :=
by sorry

end cubic_decreasing_iff_l1747_174745


namespace simplify_expression_l1747_174721

theorem simplify_expression (x : ℝ) : (3*x - 4)*(2*x + 10) - (x + 3)*(3*x - 2) = 3*x^2 + 15*x - 34 := by
  sorry

end simplify_expression_l1747_174721


namespace exponent_simplification_l1747_174760

theorem exponent_simplification :
  3^12 * 8^12 * 3^3 * 8^8 = 24^15 * 32768 := by
  sorry

end exponent_simplification_l1747_174760


namespace new_shoes_duration_proof_l1747_174798

/-- The duration of new shoes in years -/
def new_shoes_duration : ℝ := 2

/-- The cost of repairing used shoes -/
def used_shoes_repair_cost : ℝ := 10.5

/-- The duration of used shoes after repair in years -/
def used_shoes_duration : ℝ := 1

/-- The cost of new shoes -/
def new_shoes_cost : ℝ := 30

/-- The percentage increase in average cost per year of new shoes compared to repaired used shoes -/
def cost_increase_percentage : ℝ := 42.857142857142854

theorem new_shoes_duration_proof :
  new_shoes_duration = new_shoes_cost / (used_shoes_repair_cost * (1 + cost_increase_percentage / 100)) :=
by sorry

end new_shoes_duration_proof_l1747_174798


namespace regular_polygon_right_triangles_l1747_174779

/-- Given a regular polygon with n sides, if there are 1200 ways to choose
    three vertices that form a right triangle, then n = 50. -/
theorem regular_polygon_right_triangles (n : ℕ) : n > 0 →
  (n / 2 : ℕ) * (n - 2) = 1200 → n = 50 := by sorry

end regular_polygon_right_triangles_l1747_174779


namespace bruce_bank_savings_l1747_174765

/-- The amount of money Bruce puts in the bank -/
def money_in_bank (aunt_money grandfather_money : ℕ) : ℚ :=
  (aunt_money + grandfather_money : ℚ) / 5

/-- Theorem stating the amount Bruce put in the bank -/
theorem bruce_bank_savings :
  money_in_bank 75 150 = 45 := by sorry

end bruce_bank_savings_l1747_174765


namespace second_train_speed_prove_second_train_speed_l1747_174738

/-- Calculates the speed of the second train given the conditions of the problem -/
theorem second_train_speed 
  (distance : ℝ) 
  (speed_first : ℝ) 
  (extra_distance : ℝ) : ℝ :=
  let speed_second := (3 * distance - 2 * extra_distance) / (6 * distance / speed_first - 2 * extra_distance / speed_first)
  speed_second

/-- Proves that the speed of the second train is 125/3 kmph given the problem conditions -/
theorem prove_second_train_speed :
  second_train_speed 1100 50 100 = 125/3 := by
  sorry

end second_train_speed_prove_second_train_speed_l1747_174738


namespace fraction_closest_to_longest_side_l1747_174769

-- Define the quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (angleA angleD : Real)
  (angleB angleC : Real)
  (lengthAB lengthBC lengthCD lengthDA : Real)

-- Define the function to calculate the area closest to DA
def areaClosestToDA (q : Quadrilateral) : Real := sorry

-- Define the function to calculate the total area of the quadrilateral
def totalArea (q : Quadrilateral) : Real := sorry

-- Theorem statement
theorem fraction_closest_to_longest_side 
  (q : Quadrilateral)
  (h1 : q.A = (0, 0))
  (h2 : q.B = (1, 2))
  (h3 : q.C = (3, 2))
  (h4 : q.D = (4, 0))
  (h5 : q.angleA = 75)
  (h6 : q.angleD = 75)
  (h7 : q.angleB = 105)
  (h8 : q.angleC = 105)
  (h9 : q.lengthAB = 100)
  (h10 : q.lengthBC = 150)
  (h11 : q.lengthCD = 100)
  (h12 : q.lengthDA = 150)
  : areaClosestToDA q / totalArea q = areaClosestToDA q / totalArea q := by
  sorry

end fraction_closest_to_longest_side_l1747_174769


namespace johnson_potatoes_problem_l1747_174781

theorem johnson_potatoes_problem (initial_potatoes : ℕ) (remaining_potatoes : ℕ) 
  (h1 : initial_potatoes = 300)
  (h2 : remaining_potatoes = 47) :
  ∃ (gina_potatoes : ℕ),
    gina_potatoes = 69 ∧
    initial_potatoes - remaining_potatoes = 
      gina_potatoes + 2 * gina_potatoes + 2 * gina_potatoes / 3 :=
by sorry

end johnson_potatoes_problem_l1747_174781


namespace ones_digit_of_nine_to_46_l1747_174761

theorem ones_digit_of_nine_to_46 : (9^46 : ℕ) % 10 = 1 := by
  sorry

end ones_digit_of_nine_to_46_l1747_174761


namespace kanul_cash_theorem_l1747_174759

/-- The total amount of cash Kanul had -/
def total_cash : ℝ := 5714.29

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 3000

/-- The amount spent on machinery -/
def machinery : ℝ := 1000

/-- The percentage of total cash spent -/
def percentage_spent : ℝ := 0.30

theorem kanul_cash_theorem :
  total_cash = raw_materials + machinery + percentage_spent * total_cash := by
  sorry

end kanul_cash_theorem_l1747_174759


namespace sum_of_three_numbers_l1747_174790

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 35)
  (h2 : b + c = 55)
  (h3 : c + a = 62) :
  a + b + c = 76 := by
sorry

end sum_of_three_numbers_l1747_174790


namespace total_spent_calculation_l1747_174748

/-- Calculates the total amount spent at a restaurant given the food price, sales tax rate, and tip rate. -/
def totalSpent (foodPrice : ℝ) (salesTaxRate : ℝ) (tipRate : ℝ) : ℝ :=
  let priceWithTax := foodPrice * (1 + salesTaxRate)
  let tipAmount := priceWithTax * tipRate
  priceWithTax + tipAmount

/-- Theorem stating that the total amount spent is $184.80 given the specific conditions. -/
theorem total_spent_calculation (foodPrice : ℝ) (salesTaxRate : ℝ) (tipRate : ℝ) 
    (h1 : foodPrice = 140)
    (h2 : salesTaxRate = 0.1)
    (h3 : tipRate = 0.2) : 
  totalSpent foodPrice salesTaxRate tipRate = 184.80 := by
  sorry

#eval totalSpent 140 0.1 0.2

end total_spent_calculation_l1747_174748


namespace arithmetic_progression_common_difference_l1747_174710

/-- An arithmetic progression with first term 5 and 25th term 173 has common difference 7. -/
theorem arithmetic_progression_common_difference : 
  ∀ (a : ℕ → ℝ), 
    (a 1 = 5) → 
    (a 25 = 173) → 
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) → 
    (a 2 - a 1 = 7) := by
  sorry

end arithmetic_progression_common_difference_l1747_174710


namespace can_cut_one_more_square_l1747_174783

/-- Represents a grid of cells -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square region in the grid -/
structure Square :=
  (size : ℕ)

/-- The number of 2x2 squares that can fit in a grid -/
def count_2x2_squares (g : Grid) : ℕ :=
  ((g.rows - 1) / 2) * ((g.cols - 1) / 2)

theorem can_cut_one_more_square (g : Grid) (s : Square) (n : ℕ) :
  g.rows = 29 →
  g.cols = 29 →
  s.size = 2 →
  n = 99 →
  n < count_2x2_squares g →
  ∃ (remaining : ℕ), remaining > 0 ∧ remaining = count_2x2_squares g - n :=
by sorry

end can_cut_one_more_square_l1747_174783


namespace sqrt_meaningful_range_l1747_174711

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 3) ↔ x ≥ 3 / 2 := by
  sorry

end sqrt_meaningful_range_l1747_174711


namespace complement_intersection_A_B_l1747_174737

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

-- State the theorem
theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x : ℝ | x < 4 ∨ x ≥ 10} := by sorry

end complement_intersection_A_B_l1747_174737


namespace value_of_x_l1747_174799

theorem value_of_x (x y : ℚ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := by
  sorry

end value_of_x_l1747_174799
