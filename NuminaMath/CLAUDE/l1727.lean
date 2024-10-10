import Mathlib

namespace walking_scenario_l1727_172767

def distance_between_people (initial_distance : ℝ) (person1_movement : ℝ) (person2_movement : ℝ) : ℝ :=
  initial_distance + person1_movement - person2_movement

theorem walking_scenario (initial_distance : ℝ) (person1_movement : ℝ) (person2_movement : ℝ) :
  initial_distance = 400 ∧ person1_movement = 200 ∧ person2_movement = -200 →
  distance_between_people initial_distance person1_movement person2_movement = 400 :=
by
  sorry

#check walking_scenario

end walking_scenario_l1727_172767


namespace smaller_number_l1727_172792

theorem smaller_number (x y : ℝ) (sum_eq : x + y = 30) (diff_eq : x - y = 10) : 
  min x y = 10 := by
  sorry

end smaller_number_l1727_172792


namespace pet_ratio_l1727_172718

theorem pet_ratio (dogs : ℕ) (cats : ℕ) (total_pets : ℕ) : 
  dogs = 2 → cats = 3 → total_pets = 15 → 
  (total_pets - (dogs + cats)) * 1 = 2 * (dogs + cats) := by
  sorry

end pet_ratio_l1727_172718


namespace triangle_altitude_segment_l1727_172798

/-- Given a triangle with sides 35, 85, and 90 units, prove that when an altitude is dropped on the side of length 90, the length of the larger segment cut off by the altitude is 78.33 units. -/
theorem triangle_altitude_segment (a b c : ℝ) (h1 : a = 35) (h2 : b = 85) (h3 : c = 90) :
  let x := (c^2 + a^2 - b^2) / (2 * c)
  c - x = 78.33 := by sorry

end triangle_altitude_segment_l1727_172798


namespace situp_ratio_l1727_172785

/-- The number of sit-ups Ken can do -/
def ken_situps : ℕ := 20

/-- The number of sit-ups Nathan can do -/
def nathan_situps : ℕ := 2 * ken_situps

/-- The number of sit-ups Bob can do -/
def bob_situps : ℕ := ken_situps + 10

/-- The combined number of sit-ups Ken and Nathan can do -/
def ken_nathan_combined : ℕ := ken_situps + nathan_situps

theorem situp_ratio : 
  (bob_situps : ℚ) / ken_nathan_combined = 1 / 2 := by
  sorry

end situp_ratio_l1727_172785


namespace abs_z_equals_five_l1727_172762

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := sorry

-- State the theorem
theorem abs_z_equals_five :
  z * i^2018 = 3 + 4*i → Complex.abs z = 5 := by sorry

end abs_z_equals_five_l1727_172762


namespace no_valid_base_solution_l1727_172735

theorem no_valid_base_solution : 
  ¬∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ 
    (4 * x + 9 = 4 * y + 1) ∧ 
    (4 * x^2 + 7 * x + 7 = 3 * y^2 + 2 * y + 9) :=
by sorry

end no_valid_base_solution_l1727_172735


namespace correct_transformation_l1727_172753

theorem correct_transformation (y : ℝ) : 
  (|y + 1| / 2 = |y| / 3 - |3*y - 1| / 6 - y) ↔ 
  (3*y + 3 = 2*y - 3*y + 1 - 6*y) := by
sorry

end correct_transformation_l1727_172753


namespace arithmetic_sequence_properties_l1727_172752

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a3_eq_6 : a 3 = 6
  S3_eq_12 : S 3 = 12

/-- The theorem stating properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 2 * n) ∧
  (∀ n, seq.S n = n * (n + 1)) :=
sorry

end arithmetic_sequence_properties_l1727_172752


namespace astronomy_collections_l1727_172758

/-- Represents the distinct letters in "ASTRONOMY" --/
inductive AstronomyLetter
| A
| O
| S
| T
| R
| N
| M
| Y

/-- The number of each letter in "ASTRONOMY" --/
def letter_count : AstronomyLetter → Nat
| AstronomyLetter.A => 1
| AstronomyLetter.O => 2
| AstronomyLetter.S => 1
| AstronomyLetter.T => 1
| AstronomyLetter.R => 1
| AstronomyLetter.N => 2
| AstronomyLetter.M => 1
| AstronomyLetter.Y => 1

/-- The set of vowels in "ASTRONOMY" --/
def vowels : Set AstronomyLetter := {AstronomyLetter.A, AstronomyLetter.O}

/-- The set of consonants in "ASTRONOMY" --/
def consonants : Set AstronomyLetter := {AstronomyLetter.S, AstronomyLetter.T, AstronomyLetter.R, AstronomyLetter.N, AstronomyLetter.M, AstronomyLetter.Y}

/-- The number of distinct ways to choose 3 vowels and 3 consonants from "ASTRONOMY" --/
def distinct_collections : Nat := 100

theorem astronomy_collections :
  distinct_collections = 100 := by sorry


end astronomy_collections_l1727_172758


namespace cost_per_serving_is_50_cents_l1727_172746

/-- Calculates the cost per serving of mixed nuts in cents after applying a coupon -/
def cost_per_serving (original_price : ℚ) (bag_size : ℚ) (coupon_value : ℚ) (serving_size : ℚ) : ℚ :=
  ((original_price - coupon_value) / bag_size) * serving_size * 100

/-- Proves that the cost per serving of mixed nuts is 50 cents after applying the coupon -/
theorem cost_per_serving_is_50_cents :
  cost_per_serving 25 40 5 1 = 50 := by
  sorry

end cost_per_serving_is_50_cents_l1727_172746


namespace star_theorem_l1727_172716

/-- The star operation for real numbers -/
def star (a b : ℝ) : ℝ := (a - b) ^ 3

/-- Theorem: For real numbers x and y, (x-y)^3 ⋆ (y-x)^3 = 8(x-y)^9 -/
theorem star_theorem (x y : ℝ) : star ((x - y) ^ 3) ((y - x) ^ 3) = 8 * (x - y) ^ 9 := by
  sorry

end star_theorem_l1727_172716


namespace degree_of_derivative_P_l1727_172730

/-- The polynomial we are working with -/
def P (x : ℝ) : ℝ := (x^2 + 1)^5 * (x^4 + 1)^2

/-- The degree of a polynomial -/
noncomputable def degree (p : ℝ → ℝ) : ℕ := sorry

/-- The derivative of a polynomial -/
noncomputable def derivative (p : ℝ → ℝ) : ℝ → ℝ := sorry

theorem degree_of_derivative_P :
  degree (derivative P) = 17 := by sorry

end degree_of_derivative_P_l1727_172730


namespace f_at_zero_equals_four_l1727_172765

/-- Given a function f(x) = a * sin(x) + b * (x^(1/3)) + 4 where a and b are real numbers,
    prove that f(0) = 4 -/
theorem f_at_zero_equals_four (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin x + b * Real.rpow x (1/3) + 4
  f 0 = 4 := by sorry

end f_at_zero_equals_four_l1727_172765


namespace phd_total_time_l1727_172761

def phd_timeline (acclimation_time : ℝ) (basics_time : ℝ) (research_ratio : ℝ) (dissertation_ratio : ℝ) : ℝ :=
  let research_time := basics_time * (1 + research_ratio)
  let dissertation_time := acclimation_time * dissertation_ratio
  acclimation_time + basics_time + research_time + dissertation_time

theorem phd_total_time :
  phd_timeline 1 2 0.75 0.5 = 7 := by
  sorry

end phd_total_time_l1727_172761


namespace combined_cost_apples_strawberries_l1727_172731

def total_cost : ℕ := 82
def banana_cost : ℕ := 12
def bread_cost : ℕ := 9
def milk_cost : ℕ := 7
def apple_cost : ℕ := 15
def orange_cost : ℕ := 13
def strawberry_cost : ℕ := 26

theorem combined_cost_apples_strawberries :
  apple_cost + strawberry_cost = 41 :=
by sorry

end combined_cost_apples_strawberries_l1727_172731


namespace simplify_expression_l1727_172727

theorem simplify_expression (x y : ℝ) : 8*y + 15 - 3*y + 20 + 2*x = 5*y + 2*x + 35 := by
  sorry

end simplify_expression_l1727_172727


namespace nancy_file_deletion_l1727_172715

theorem nancy_file_deletion (initial_files : ℕ) (num_folders : ℕ) (files_per_folder : ℕ) : 
  initial_files = 80 → 
  num_folders = 7 → 
  files_per_folder = 7 → 
  initial_files - (num_folders * files_per_folder) = 31 := by
sorry

end nancy_file_deletion_l1727_172715


namespace quarter_circle_roll_path_length_l1727_172784

/-- The length of the path traveled by point B when rolling a quarter-circle along another quarter-circle -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 4 / π) :
  let path_length := 2 * π * r
  path_length = 8 := by sorry

end quarter_circle_roll_path_length_l1727_172784


namespace sum_of_roots_l1727_172724

theorem sum_of_roots (a b : ℝ) : 
  a ≠ b → a * (a - 4) = 12 → b * (b - 4) = 12 → a + b = 4 := by
  sorry

end sum_of_roots_l1727_172724


namespace sum_of_squares_perfect_square_l1727_172777

theorem sum_of_squares_perfect_square (n p k : ℤ) :
  ∃ m : ℤ, n^2 + p^2 + k^2 = m^2 ↔ n * k = (p / 2)^2 := by
sorry

end sum_of_squares_perfect_square_l1727_172777


namespace smaller_number_in_ratio_l1727_172737

theorem smaller_number_in_ratio (x y d : ℝ) : 
  x > 0 → y > 0 → x / y = 2 / 3 → 2 * x + 3 * y = d → min x y = 2 * d / 13 := by
  sorry

end smaller_number_in_ratio_l1727_172737


namespace lcm_12_21_30_l1727_172729

theorem lcm_12_21_30 : Nat.lcm (Nat.lcm 12 21) 30 = 420 := by
  sorry

end lcm_12_21_30_l1727_172729


namespace shekar_science_marks_l1727_172722

/-- Calculates the marks in science given other subject marks and the average -/
def calculate_science_marks (math social english biology average : ℕ) : ℕ :=
  5 * average - (math + social + english + biology)

/-- Proves that Shekar's science marks are 65 given his other marks and average -/
theorem shekar_science_marks :
  let math := 76
  let social := 82
  let english := 62
  let biology := 85
  let average := 74
  calculate_science_marks math social english biology average = 65 := by
  sorry

#eval calculate_science_marks 76 82 62 85 74

end shekar_science_marks_l1727_172722


namespace equality_of_fractions_l1727_172743

theorem equality_of_fractions (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 49 ∧ (4 : ℚ) / 7 = 84 / N → M - N = -119 := by
  sorry

end equality_of_fractions_l1727_172743


namespace parabola_focus_directrix_distance_l1727_172714

/-- Given a parabola with equation x² = 8y, the distance from its focus to its directrix is 4. -/
theorem parabola_focus_directrix_distance : 
  ∀ (x y : ℝ), x^2 = 8*y → (∃ (focus_distance : ℝ), focus_distance = 4) :=
by sorry

end parabola_focus_directrix_distance_l1727_172714


namespace second_player_loses_l1727_172738

/-- Represents the game state -/
structure GameState :=
  (diamonds : ℕ)

/-- Represents a move in the game -/
def is_valid_move (s : GameState) (s' : GameState) : Prop :=
  s'.diamonds = s.diamonds + 1 ∧ s'.diamonds ≤ 2017

/-- The game ends when there are 2017 piles (diamonds) -/
def game_over (s : GameState) : Prop :=
  s.diamonds = 2017

/-- The number of moves required to finish the game -/
def moves_to_end (s : GameState) : ℕ :=
  2017 - s.diamonds

/-- Theorem: The second player loses in a game starting with 2017 diamonds -/
theorem second_player_loses :
  ∀ (s : GameState),
    s.diamonds = 1 →
    ∃ (strategy : GameState → GameState),
      (∀ (s' : GameState), is_valid_move s s' → is_valid_move s' (strategy s')) →
      (moves_to_end s) % 2 = 0 →
      game_over (strategy s) :=
sorry

end second_player_loses_l1727_172738


namespace max_abs_z_value_l1727_172781

theorem max_abs_z_value (a b c z : ℂ) (d : ℝ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs a = (1 / 2) * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + d * c = 0)
  (h5 : d = 1) :
  ∃ (M : ℝ), M = 2 ∧ ∀ z', a * z'^2 + b * z' + d * c = 0 → Complex.abs z' ≤ M :=
sorry

end max_abs_z_value_l1727_172781


namespace polynomial_inequality_l1727_172773

theorem polynomial_inequality (x : ℝ) : 1 + x^2 + x^6 + x^8 ≥ 4 * x^4 := by
  sorry

end polynomial_inequality_l1727_172773


namespace exp_properties_l1727_172736

-- Define the exponential function as a power series
noncomputable def Exp (z : ℂ) : ℂ := ∑' n, z^n / n.factorial

-- State the properties to be proven
theorem exp_properties :
  (∀ z : ℂ, HasDerivAt Exp (Exp z) z) ∧
  (∀ (α β : ℝ) (z : ℂ), Exp ((α + β) • z) = Exp (α • z) * Exp (β • z)) := by
  sorry

end exp_properties_l1727_172736


namespace equation_solution_l1727_172703

theorem equation_solution : ∃ x : ℚ, 5 * (x - 8) + 6 = 3 * (3 - 3 * x) + 15 ∧ x = 29 / 7 := by
  sorry

end equation_solution_l1727_172703


namespace parallelogram_area_48_36_l1727_172726

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 48 cm and height 36 cm is 1728 square centimeters -/
theorem parallelogram_area_48_36 : parallelogram_area 48 36 = 1728 := by
  sorry

end parallelogram_area_48_36_l1727_172726


namespace negative_product_plus_two_l1727_172717

theorem negative_product_plus_two :
  ∀ (a b : ℤ), a = -2 → b = -3 → a * b + 2 = 8 := by
  sorry

end negative_product_plus_two_l1727_172717


namespace at_least_five_primes_in_cubic_l1727_172794

theorem at_least_five_primes_in_cubic (f : ℕ → ℕ) : 
  (∀ n : ℕ, f n = n^3 - 10*n^2 + 31*n - 17) →
  ∃ (a b c d e : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    Nat.Prime (f a) ∧ Nat.Prime (f b) ∧ Nat.Prime (f c) ∧ Nat.Prime (f d) ∧ Nat.Prime (f e) :=
by sorry

end at_least_five_primes_in_cubic_l1727_172794


namespace quadratic_root_implies_m_l1727_172793

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 1) → 
  (1^2 - 2*1 + m = 1) → 
  m = 2 := by sorry

end quadratic_root_implies_m_l1727_172793


namespace sin_30_degrees_l1727_172712

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end sin_30_degrees_l1727_172712


namespace max_value_on_ellipse_l1727_172709

/-- The ellipse defined by 2x^2 + 3y^2 = 12 -/
def Ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 12

/-- The function to be maximized -/
def f (x y : ℝ) : ℝ := x + 2 * y

/-- Theorem stating that the maximum value of x + 2y on the given ellipse is √22 -/
theorem max_value_on_ellipse :
  ∃ (max : ℝ), max = Real.sqrt 22 ∧
  (∀ x y : ℝ, Ellipse x y → f x y ≤ max) ∧
  (∃ x y : ℝ, Ellipse x y ∧ f x y = max) := by
  sorry

end max_value_on_ellipse_l1727_172709


namespace S_6_value_l1727_172783

theorem S_6_value (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 := by
  sorry

end S_6_value_l1727_172783


namespace constant_term_value_l1727_172790

/-- The binomial expansion of (x - 2/x)^8 has its maximum coefficient in the 5th term -/
def max_coeff_5th_term (x : ℝ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ 8 → Nat.choose 8 4 ≥ Nat.choose 8 k

/-- The constant term in the binomial expansion of (x - 2/x)^8 -/
def constant_term (x : ℝ) : ℤ :=
  Nat.choose 8 4 * (-2)^4

theorem constant_term_value (x : ℝ) :
  max_coeff_5th_term x → constant_term x = 1120 := by
  sorry

end constant_term_value_l1727_172790


namespace range_of_a_l1727_172721

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a^2 * x - 2 ≤ 0) → -2 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l1727_172721


namespace sum_bound_l1727_172763

theorem sum_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = 1) :
  a + b > 9 ∧ ∀ ε > 0, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1/a' + 4/b' = 1 ∧ a' + b' < 9 + ε :=
sorry

end sum_bound_l1727_172763


namespace min_breaks_for_40_tiles_l1727_172751

/-- Represents a chocolate bar -/
structure ChocolateBar where
  tiles : ℕ

/-- Represents the breaking process -/
def breakChocolate (initial : ChocolateBar) (breaks : ℕ) : ℕ :=
  initial.tiles + breaks

/-- Theorem: The minimum number of breaks required for a 40-tile chocolate bar is 39 -/
theorem min_breaks_for_40_tiles (bar : ChocolateBar) (h : bar.tiles = 40) :
  ∃ (breaks : ℕ), breakChocolate bar breaks = 40 ∧ 
  ∀ (n : ℕ), breakChocolate bar n = 40 → breaks ≤ n :=
by sorry

end min_breaks_for_40_tiles_l1727_172751


namespace field_width_l1727_172705

/-- The width of a rectangular field satisfying specific conditions -/
theorem field_width : ∃ (W : ℝ), W = 10 ∧ 
  20 * W * 0.5 - 40 * 0.5 = 8 * 5 * 2 := by
  sorry

end field_width_l1727_172705


namespace omega_circle_l1727_172702

open Complex

/-- Given a complex number z satisfying |z - i| = 1, z ≠ 0, z ≠ 2i, and a complex number ω
    such that (ω / (ω - 2i)) * ((z - 2i) / z) is real, prove that ω lies on the circle
    centered at (0, 1) with radius 1, excluding the points (0, 0) and (0, 2). -/
theorem omega_circle (z ω : ℂ) 
  (h1 : abs (z - I) = 1)
  (h2 : z ≠ 0)
  (h3 : z ≠ 2 * I)
  (h4 : ∃ (r : ℝ), ω / (ω - 2 * I) * ((z - 2 * I) / z) = r) :
  abs (ω - I) = 1 ∧ ω ≠ 0 ∧ ω ≠ 2 * I :=
sorry

end omega_circle_l1727_172702


namespace additional_profit_special_house_l1727_172766

/-- The selling price of standard houses in the area -/
def standard_house_price : ℝ := 320000

/-- The additional cost to build the special house -/
def additional_build_cost : ℝ := 100000

/-- The factor by which the special house sells compared to standard houses -/
def special_house_price_factor : ℝ := 1.5

/-- Theorem stating the additional profit made by building the special house -/
theorem additional_profit_special_house : 
  (special_house_price_factor * standard_house_price - standard_house_price) - additional_build_cost = 60000 := by
  sorry

end additional_profit_special_house_l1727_172766


namespace kolya_twos_count_l1727_172734

/-- Represents the grades of a student -/
structure Grades where
  fives : ℕ
  fours : ℕ
  threes : ℕ
  twos : ℕ

/-- Calculates the average grade -/
def averageGrade (g : Grades) : ℚ :=
  (5 * g.fives + 4 * g.fours + 3 * g.threes + 2 * g.twos) / 20

theorem kolya_twos_count 
  (kolya vasya : Grades)
  (total_grades : kolya.fives + kolya.fours + kolya.threes + kolya.twos = 20)
  (vasya_total : vasya.fives + vasya.fours + vasya.threes + vasya.twos = 20)
  (fives_eq : kolya.fives = vasya.fours)
  (fours_eq : kolya.fours = vasya.threes)
  (threes_eq : kolya.threes = vasya.twos)
  (twos_eq : kolya.twos = vasya.fives)
  (avg_eq : averageGrade kolya = averageGrade vasya) :
  kolya.twos = 5 := by
sorry

end kolya_twos_count_l1727_172734


namespace largest_valid_number_l1727_172747

def is_valid_number (n : ℕ) : Prop :=
  ∃ (r : ℕ) (i : ℕ), 
    i > 0 ∧ 
    i < (Nat.digits 10 n).length ∧ 
    n % 10 ≠ 0 ∧
    r > 1 ∧
    r * (n / 10^(i + 1) * 10^i + n % 10^i) = n

theorem largest_valid_number : 
  is_valid_number 180625 ∧ 
  ∀ m : ℕ, m > 180625 → ¬(is_valid_number m) :=
sorry

end largest_valid_number_l1727_172747


namespace polynomial_factorization_l1727_172795

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 1) * (x^2 + 6*x + 37) := by
  sorry

end polynomial_factorization_l1727_172795


namespace absolute_value_equation_solution_l1727_172797

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x - 5| = 3 * x + 1 ↔ x = 1 := by
  sorry

end absolute_value_equation_solution_l1727_172797


namespace distinct_prime_factors_of_30_factorial_l1727_172741

theorem distinct_prime_factors_of_30_factorial (n : ℕ) :
  n = 30 →
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card = 
  (Finset.filter (λ p => p.Prime ∧ p ∣ n.factorial) (Finset.range (n + 1))).card :=
by sorry

end distinct_prime_factors_of_30_factorial_l1727_172741


namespace green_balls_removal_l1727_172787

theorem green_balls_removal (total : ℕ) (red_percent : ℚ) (green_removed : ℕ) :
  total = 150 →
  red_percent = 2/5 →
  green_removed = 75 →
  (red_percent * ↑total : ℚ) / (↑total - ↑green_removed : ℚ) = 4/5 :=
by sorry

end green_balls_removal_l1727_172787


namespace assignments_for_thirty_points_l1727_172719

/-- Calculates the number of assignments needed for a given number of points -/
def assignments_needed (points : ℕ) : ℕ :=
  if points ≤ 10 then points
  else if points ≤ 20 then 10 + 2 * (points - 10)
  else 30 + 3 * (points - 20)

/-- Theorem stating that 60 assignments are needed for 30 points -/
theorem assignments_for_thirty_points :
  assignments_needed 30 = 60 := by sorry

end assignments_for_thirty_points_l1727_172719


namespace dress_cost_theorem_l1727_172707

/-- The cost of a dress given the initial and remaining number of quarters -/
def dress_cost (initial_quarters remaining_quarters : ℕ) : ℚ :=
  (initial_quarters - remaining_quarters) * (1 / 4)

/-- Theorem stating that the dress cost $35 given the initial and remaining quarters -/
theorem dress_cost_theorem (initial_quarters remaining_quarters : ℕ) 
  (h1 : initial_quarters = 160)
  (h2 : remaining_quarters = 20) :
  dress_cost initial_quarters remaining_quarters = 35 := by
  sorry

#eval dress_cost 160 20

end dress_cost_theorem_l1727_172707


namespace cube_volume_in_box_l1727_172750

/-- Given a box with dimensions 9 cm × 12 cm × 3 cm, filled with 108 identical cubes,
    the volume of each cube is 27 cm³. -/
theorem cube_volume_in_box (length width height : ℕ) (num_cubes : ℕ) :
  length = 9 ∧ width = 12 ∧ height = 3 ∧ num_cubes = 108 →
  ∃ (cube_volume : ℕ), cube_volume = 27 ∧ num_cubes * cube_volume = length * width * height :=
by sorry

end cube_volume_in_box_l1727_172750


namespace smallest_four_digit_sum_15_l1727_172700

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_four_digit_sum_15 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 15 → n ≥ 1009 :=
by sorry

end smallest_four_digit_sum_15_l1727_172700


namespace sqrt_50_simplified_l1727_172759

theorem sqrt_50_simplified : Real.sqrt 50 = 5 * Real.sqrt 2 := by
  sorry

end sqrt_50_simplified_l1727_172759


namespace negation_of_implication_l1727_172757

theorem negation_of_implication (A B : Set α) :
  ¬(∀ a, a ∈ A → b ∈ B) ↔ ∃ a, a ∈ A ∧ b ∉ B :=
by sorry

end negation_of_implication_l1727_172757


namespace election_winner_percentage_l1727_172701

theorem election_winner_percentage (total_votes winner_votes margin : ℕ) : 
  winner_votes = 720 →
  margin = 240 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 3/5 := by
sorry

end election_winner_percentage_l1727_172701


namespace folded_square_area_l1727_172748

/-- The area of a shape formed by folding a square along its diagonal -/
theorem folded_square_area (side_length : ℝ) (h : side_length = 2) : 
  (side_length ^ 2) / 2 = 2 := by sorry

end folded_square_area_l1727_172748


namespace total_cost_calculation_l1727_172768

/-- The total cost of sandwiches and sodas -/
def total_cost (sandwich_quantity : ℕ) (sandwich_price : ℚ) (soda_quantity : ℕ) (soda_price : ℚ) : ℚ :=
  sandwich_quantity * sandwich_price + soda_quantity * soda_price

/-- Proof that the total cost of 2 sandwiches at $1.49 each and 4 sodas at $0.87 each is $6.46 -/
theorem total_cost_calculation : total_cost 2 (149/100) 4 (87/100) = 646/100 := by
  sorry

end total_cost_calculation_l1727_172768


namespace point_positions_l1727_172742

def line_equation (x y : ℝ) : Prop := 3 * x - 5 * y + 8 = 0

def point_A : ℝ × ℝ := (2, 5)
def point_B : ℝ × ℝ := (1, 2.2)

theorem point_positions :
  (¬ line_equation point_A.1 point_A.2) ∧
  (line_equation point_B.1 point_B.2) :=
by sorry

end point_positions_l1727_172742


namespace hockey_tournament_points_l1727_172779

/-- The number of teams in the tournament -/
def num_teams : ℕ := 2016

/-- The number of points awarded for a win -/
def points_per_win : ℕ := 3

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The total number of points awarded in the tournament -/
def total_points : ℕ := total_games * points_per_win

theorem hockey_tournament_points :
  total_points = 6093360 := by sorry

end hockey_tournament_points_l1727_172779


namespace quadratic_polynomial_condition_l1727_172733

/-- 
Given a polynomial p(x) = 2a x^4 + 5a x^3 - 13 x^2 - x^4 + 2021 + 2x + bx^3 - bx^4 - 13x^3,
if p(x) is a quadratic polynomial, then a^2 + b^2 = 13.
-/
theorem quadratic_polynomial_condition (a b : ℝ) : 
  (∀ x : ℝ, (2*a - b - 1) * x^4 + (5*a + b - 13) * x^3 - 13 * x^2 + 2 * x + 2021 = 
             -13 * x^2 + 2 * x + 2021) → 
  a^2 + b^2 = 13 := by
  sorry

end quadratic_polynomial_condition_l1727_172733


namespace relation_between_exponents_l1727_172745

-- Define variables
variable (a b c d x y p z : ℝ)

-- Define the theorem
theorem relation_between_exponents 
  (h1 : a^x = b^p) 
  (h2 : b^p = c)
  (h3 : b^y = a^z) 
  (h4 : a^z = d)
  : p * y = x * z := by
  sorry

end relation_between_exponents_l1727_172745


namespace binary_arithmetic_equality_l1727_172774

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits. -/
def binary (bits : List Bool) : Nat := binaryToNat bits

theorem binary_arithmetic_equality : 
  (binary [true, false, true, true, true, false] + binary [true, false, true, false, true]) -
  (binary [true, true, true, false, false, false] - binary [true, true, false, true, false, true]) +
  binary [true, true, true, false, true] =
  binary [true, false, true, true, true, false, true] := by
  sorry

#eval binary [true, false, true, true, true, false, true]

end binary_arithmetic_equality_l1727_172774


namespace recurrence_sequence_a1_zero_l1727_172704

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (c : ℝ) (a : ℕ → ℝ) : Prop :=
  c > 2 ∧ ∀ n : ℕ, a n = (a (n - 1))^2 - a (n - 1) ∧ a n < 1 / Real.sqrt (c * n)

/-- The main theorem stating that a₁ = 0 for any sequence satisfying the recurrence relation -/
theorem recurrence_sequence_a1_zero (c : ℝ) (a : ℕ → ℝ) (h : RecurrenceSequence c a) : a 1 = 0 := by
  sorry

end recurrence_sequence_a1_zero_l1727_172704


namespace polygon_sides_l1727_172708

/-- The number of sides of a polygon given the difference between its interior and exterior angle sums -/
theorem polygon_sides (interior_exterior_diff : ℝ) : interior_exterior_diff = 540 → ∃ n : ℕ, n = 7 ∧ 
  (n - 2) * 180 = 360 + interior_exterior_diff ∧ 
  (∀ m : ℕ, (m - 2) * 180 = 360 + interior_exterior_diff → m = n) := by
  sorry

end polygon_sides_l1727_172708


namespace max_combined_power_l1727_172796

theorem max_combined_power (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ < 1) (h₂ : x₂ < 1) (h₃ : x₃ < 1)
  (h : 2 * (x₁ + x₂ + x₃) + 4 * x₁ * x₂ * x₃ = 3 * (x₁ * x₂ + x₁ * x₃ + x₂ * x₃) + 1) :
  x₁ + x₂ + x₃ ≤ 3/4 := by
sorry

end max_combined_power_l1727_172796


namespace prob_at_least_one_3_l1727_172791

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The probability of rolling a 3 on a single fair die -/
def probThree : ℚ := 1 / numSides

/-- The probability of not rolling a 3 on a single fair die -/
def probNotThree : ℚ := 1 - probThree

/-- The probability of rolling at least one 3 when two fair dice are rolled -/
def probAtLeastOne3 : ℚ := 1 - probNotThree * probNotThree

theorem prob_at_least_one_3 : probAtLeastOne3 = 11 / 36 := by
  sorry

end prob_at_least_one_3_l1727_172791


namespace inequality_proof_l1727_172740

theorem inequality_proof (n : ℕ) (a : ℝ) (hn : n > 1) (ha : 0 < a ∧ a < 1) :
  1 + a < (1 + a / n)^n ∧ (1 + a / n)^n < (1 + a / (n + 1))^(n + 1) := by
  sorry

end inequality_proof_l1727_172740


namespace sandrine_dishes_washed_l1727_172771

def number_of_pears_picked : ℕ := 50

def number_of_bananas_cooked (pears : ℕ) : ℕ := 3 * pears

def number_of_dishes_washed (bananas : ℕ) : ℕ := bananas + 10

theorem sandrine_dishes_washed :
  number_of_dishes_washed (number_of_bananas_cooked number_of_pears_picked) = 160 := by
  sorry

end sandrine_dishes_washed_l1727_172771


namespace basic_computer_price_theorem_l1727_172786

/-- Represents the prices of computer components and setups -/
structure ComputerPrices where
  basic_total : ℝ  -- Total price of basic setup
  enhanced_total : ℝ  -- Total price of enhanced setup
  printer_ratio : ℝ  -- Ratio of printer price to enhanced total
  monitor_ratio : ℝ  -- Ratio of monitor price to enhanced total
  keyboard_ratio : ℝ  -- Ratio of keyboard price to enhanced total

/-- Calculates the price of the basic computer given the prices and ratios -/
def basic_computer_price (prices : ComputerPrices) : ℝ :=
  let enhanced_computer := prices.enhanced_total * (1 - prices.printer_ratio - prices.monitor_ratio - prices.keyboard_ratio)
  enhanced_computer - (prices.enhanced_total - prices.basic_total)

/-- Theorem stating that the basic computer price is approximately $975.83 -/
theorem basic_computer_price_theorem (prices : ComputerPrices) 
  (h1 : prices.basic_total = 2500)
  (h2 : prices.enhanced_total = prices.basic_total + 600)
  (h3 : prices.printer_ratio = 1/6)
  (h4 : prices.monitor_ratio = 1/5)
  (h5 : prices.keyboard_ratio = 1/8) :
  ∃ ε > 0, |basic_computer_price prices - 975.83| < ε :=
sorry

end basic_computer_price_theorem_l1727_172786


namespace picture_area_is_6600_l1727_172720

/-- Calculates the area of a picture within a rectangular frame. -/
def picture_area (outer_height outer_width short_frame_width long_frame_width : ℕ) : ℕ :=
  (outer_height - 2 * short_frame_width) * (outer_width - 2 * long_frame_width)

/-- Theorem stating that for a frame with given dimensions, the enclosed picture has an area of 6600 cm². -/
theorem picture_area_is_6600 :
  picture_area 100 140 20 15 = 6600 := by
  sorry

end picture_area_is_6600_l1727_172720


namespace intersection_A_B_empty_complement_A_union_B_a_geq_one_l1727_172756

-- Define the sets A, B, U, and C
def A : Set ℝ := {x | x^2 + 6*x + 5 < 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 1}
def U : Set ℝ := {x | |x| < 5}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1: A ∩ B = ∅
theorem intersection_A_B_empty : A ∩ B = ∅ := by sorry

-- Theorem 2: ∁_U(A ∪ B) = {x | 1 ≤ x < 5}
theorem complement_A_union_B : 
  (A ∪ B)ᶜ ∩ U = {x : ℝ | 1 ≤ x ∧ x < 5} := by sorry

-- Theorem 3: B ∩ C = B implies a ≥ 1
theorem a_geq_one (a : ℝ) (h : B ∩ C a = B) : a ≥ 1 := by sorry

end intersection_A_B_empty_complement_A_union_B_a_geq_one_l1727_172756


namespace petrol_price_increase_l1727_172769

theorem petrol_price_increase (original_price : ℝ) (original_consumption : ℝ) : 
  let consumption_reduction : ℝ := 0.2857142857142857
  let new_consumption : ℝ := original_consumption * (1 - consumption_reduction)
  let new_price : ℝ := original_price * original_consumption / new_consumption
  new_price / original_price - 1 = 0.4 := by sorry

end petrol_price_increase_l1727_172769


namespace bus_capacity_proof_l1727_172782

theorem bus_capacity_proof (C : ℕ) : 
  (3 : ℚ) / 5 * C + 32 = C → C = 80 := by
  sorry

end bus_capacity_proof_l1727_172782


namespace z_pure_imaginary_iff_z_in_fourth_quadrant_iff_l1727_172780

def z (m : ℝ) : ℂ := (1 + Complex.I) * m^2 - 3 * Complex.I * m + 2 * Complex.I - 1

theorem z_pure_imaginary_iff (m : ℝ) : 
  z m = Complex.I * Complex.im (z m) ↔ m = -1 :=
sorry

theorem z_in_fourth_quadrant_iff (m : ℝ) :
  Complex.re (z m) > 0 ∧ Complex.im (z m) < 0 ↔ 1 < m ∧ m < 2 :=
sorry

end z_pure_imaginary_iff_z_in_fourth_quadrant_iff_l1727_172780


namespace inequality_solution_set_l1727_172778

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x - 1) - 5 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 20) ↔ 
  (x < -1 ∨ (2 < x ∧ x < 3) ∨ (5 < x ∧ x < 6)) :=
by sorry

end inequality_solution_set_l1727_172778


namespace simplify_fraction_l1727_172732

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l1727_172732


namespace function_always_negative_iff_m_in_range_l1727_172776

/-- The function f(x) = mx^2 - mx - 1 is negative for all real x
    if and only if m is in the interval (-4, 0]. -/
theorem function_always_negative_iff_m_in_range (m : ℝ) :
  (∀ x, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by sorry

end function_always_negative_iff_m_in_range_l1727_172776


namespace long_knight_min_moves_l1727_172772

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents a move of the long knight -/
inductive LongKnightMove
  | horizontal : Bool → LongKnightMove  -- True for right, False for left
  | vertical : Bool → LongKnightMove    -- True for up, False for down

/-- The size of the chessboard -/
def boardSize : Nat := 8

/-- Applies a long knight move to a position -/
def applyMove (pos : Position) (move : LongKnightMove) : Position :=
  match move with
  | LongKnightMove.horizontal right =>
      let newX := if right then min (pos.x + 3) (boardSize - 1) else max (pos.x - 3) 0
      let newY := if right then min (pos.y + 1) (boardSize - 1) else max (pos.y - 1) 0
      ⟨newX, newY⟩
  | LongKnightMove.vertical up =>
      let newX := if up then min (pos.x + 1) (boardSize - 1) else max (pos.x - 1) 0
      let newY := if up then min (pos.y + 3) (boardSize - 1) else max (pos.y - 3) 0
      ⟨newX, newY⟩

/-- Checks if a position is at the opposite corner -/
def isOppositeCorner (pos : Position) : Prop :=
  pos.x = boardSize - 1 ∧ pos.y = boardSize - 1

/-- Theorem: The minimum number of moves for a long knight to reach the opposite corner is 5 -/
theorem long_knight_min_moves :
  ∀ (moves : List LongKnightMove),
    let finalPos := moves.foldl applyMove ⟨0, 0⟩
    isOppositeCorner finalPos → moves.length ≥ 5 :=
sorry

end long_knight_min_moves_l1727_172772


namespace complex_fraction_sum_complex_equation_solution_l1727_172710

-- Define the complex number i
def i : ℂ := Complex.I

-- Problem 1
theorem complex_fraction_sum : 
  (1 + i)^2 / (1 + 2*i) + (1 - i)^2 / (2 - i) = 6/5 - 2/5 * i := by sorry

-- Problem 2
theorem complex_equation_solution (x y : ℝ) :
  x / (1 + i) + y / (1 + 2*i) = 10 / (1 + 3*i) → x = -2 ∧ y = 10 := by sorry

end complex_fraction_sum_complex_equation_solution_l1727_172710


namespace circle_diameter_l1727_172713

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 :=
by sorry

end circle_diameter_l1727_172713


namespace derived_point_relation_find_original_point_translated_derived_point_l1727_172749

/-- Definition of an a-th order derived point -/
def derived_point (a : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (a * P.1 + P.2, P.1 + a * P.2)

/-- Theorem stating the relationship between a point and its a-th order derived point -/
theorem derived_point_relation (a : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  a ≠ 0 → Q = derived_point a P ↔ 
    Q.1 = a * P.1 + P.2 ∧ Q.2 = P.1 + a * P.2 := by
  sorry

/-- Theorem for finding the original point given its a-th order derived point -/
theorem find_original_point (a : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  a ≠ 0 ∧ a ≠ 1 → Q = derived_point a P →
    P = ((a * Q.2 - Q.1) / (a * a - 1), (a * Q.1 - Q.2) / (a * a - 1)) := by
  sorry

/-- Theorem for the composition of translation and derived point transformation -/
theorem translated_derived_point (a c : ℝ) :
  let P : ℝ × ℝ := (c + 1, 2 * c - 1)
  let P₁ : ℝ × ℝ := (c - 1, 2 * c)
  let P₂ : ℝ × ℝ := derived_point (-3) P₁
  (P₂.1 = 0 ∨ P₂.2 = 0) →
    (P₂ = (0, -16) ∨ P₂ = (16/5, 0)) := by
  sorry

end derived_point_relation_find_original_point_translated_derived_point_l1727_172749


namespace balloon_height_is_9482_l1727_172788

/-- Calculates the maximum height a balloon can fly given the following parameters:
    * total_money: The total amount of money available
    * sheet_cost: The cost of the balloon sheet
    * rope_cost: The cost of the rope
    * propane_cost: The cost of the propane tank and burner
    * helium_cost_per_oz: The cost of helium per ounce
    * height_per_oz: The height gain per ounce of helium
-/
def max_balloon_height (total_money sheet_cost rope_cost propane_cost helium_cost_per_oz height_per_oz : ℚ) : ℚ :=
  let remaining_money := total_money - (sheet_cost + rope_cost + propane_cost)
  let helium_oz := remaining_money / helium_cost_per_oz
  helium_oz * height_per_oz

/-- Theorem stating that given the specific conditions in the problem,
    the maximum height the balloon can fly is 9482 feet. -/
theorem balloon_height_is_9482 :
  max_balloon_height 200 42 18 14 1.5 113 = 9482 := by
  sorry

end balloon_height_is_9482_l1727_172788


namespace decimal_multiplication_l1727_172760

theorem decimal_multiplication (a b : ℕ) (h : a * b = 19732) :
  (a : ℚ) / 100 * ((b : ℚ) / 100) = 1.9732 :=
by sorry

end decimal_multiplication_l1727_172760


namespace z_purely_imaginary_z_in_fourth_quadrant_l1727_172725

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m * (m + 2)) (m^2 + m - 2)

-- Part 1: z is purely imaginary iff m = 0
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * Complex.im (z m) ↔ m = 0 :=
sorry

-- Part 2: z is in the fourth quadrant iff 0 < m < 1
theorem z_in_fourth_quadrant (m : ℝ) : 
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0) ↔ (0 < m ∧ m < 1) :=
sorry

end z_purely_imaginary_z_in_fourth_quadrant_l1727_172725


namespace camp_food_ratio_l1727_172728

/-- The ratio of food eaten by a dog to a puppy -/
def food_ratio (num_puppies num_dogs : ℕ) 
               (puppy_meal_frequency dog_meal_frequency : ℕ) 
               (dog_food_per_meal : ℝ) 
               (total_food_per_day : ℝ) : ℚ := by
  -- Define the ratio of food eaten by a dog to a puppy
  sorry

/-- Theorem stating the food ratio given the problem conditions -/
theorem camp_food_ratio : 
  food_ratio 4 3 9 3 4 108 = 2 := by
  sorry

end camp_food_ratio_l1727_172728


namespace zero_in_A_l1727_172711

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : (0 : ℝ) ∈ A := by
  sorry

end zero_in_A_l1727_172711


namespace min_value_xy_l1727_172755

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_geom : ∃ r : ℝ, (Real.log x) * r = (1/2) ∧ (1/2) * r = Real.log y) : 
  (∀ a b : ℝ, a > 1 → b > 1 → 
    (∃ r : ℝ, (Real.log a) * r = (1/2) ∧ (1/2) * r = Real.log b) → 
    x * y ≤ a * b) → 
  x * y = Real.exp 1 :=
sorry

end min_value_xy_l1727_172755


namespace assembly_line_production_rate_l1727_172789

theorem assembly_line_production_rate 
  (initial_rate : ℝ) 
  (initial_order : ℝ) 
  (second_order : ℝ) 
  (average_output : ℝ) 
  (h1 : initial_rate = 90) 
  (h2 : initial_order = 60) 
  (h3 : second_order = 60) 
  (h4 : average_output = 72) : 
  ∃ (reduced_rate : ℝ), 
    reduced_rate = 60 ∧ 
    (initial_order / initial_rate + second_order / reduced_rate) * average_output = initial_order + second_order :=
by sorry

end assembly_line_production_rate_l1727_172789


namespace meet_twice_l1727_172764

/-- Represents the meeting scenario between Michael and the garbage truck -/
structure MeetingScenario where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of times Michael and the truck meet -/
def number_of_meetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- The theorem stating that Michael and the truck meet exactly twice -/
theorem meet_twice (scenario : MeetingScenario) 
  (h1 : scenario.michael_speed = 6)
  (h2 : scenario.truck_speed = 12)
  (h3 : scenario.pail_distance = 240)
  (h4 : scenario.truck_stop_time = 40)
  (h5 : scenario.initial_distance = 240) :
  number_of_meetings scenario = 2 :=
sorry

end meet_twice_l1727_172764


namespace hcf_of_ratio_and_lcm_l1727_172723

/-- 
Given three positive integers a, b, and c that are in the ratio 3:4:5 and 
have a least common multiple of 2400, prove that their highest common factor is 20.
-/
theorem hcf_of_ratio_and_lcm (a b c : ℕ+) 
  (h_ratio : ∃ (k : ℕ+), a = 3 * k ∧ b = 4 * k ∧ c = 5 * k)
  (h_lcm : Nat.lcm a (Nat.lcm b c) = 2400) :
  Nat.gcd a (Nat.gcd b c) = 20 := by
  sorry

end hcf_of_ratio_and_lcm_l1727_172723


namespace charity_ticket_revenue_l1727_172770

theorem charity_ticket_revenue :
  ∀ (full_price_tickets half_price_tickets : ℕ) (full_price : ℕ),
    full_price_tickets + half_price_tickets = 180 →
    full_price_tickets * full_price + half_price_tickets * (full_price / 2) = 2750 →
    full_price_tickets * full_price = 1000 :=
by
  sorry

end charity_ticket_revenue_l1727_172770


namespace tom_buys_four_papayas_l1727_172799

/-- Represents the fruit purchase scenario --/
structure FruitPurchase where
  lemon_price : ℕ
  papaya_price : ℕ
  mango_price : ℕ
  discount_threshold : ℕ
  discount_amount : ℕ
  lemons_bought : ℕ
  mangos_bought : ℕ
  total_paid : ℕ

/-- Calculates the number of papayas bought --/
def papayas_bought (fp : FruitPurchase) (p : ℕ) : Prop :=
  let total_fruits := fp.lemons_bought + fp.mangos_bought + p
  let total_cost := fp.lemon_price * fp.lemons_bought + 
                    fp.papaya_price * p + 
                    fp.mango_price * fp.mangos_bought
  let discount := (total_fruits / fp.discount_threshold) * fp.discount_amount
  total_cost - discount = fp.total_paid

/-- Theorem stating that Tom buys 4 papayas --/
theorem tom_buys_four_papayas : 
  ∃ (fp : FruitPurchase), 
    fp.lemon_price = 2 ∧ 
    fp.papaya_price = 1 ∧ 
    fp.mango_price = 4 ∧ 
    fp.discount_threshold = 4 ∧ 
    fp.discount_amount = 1 ∧ 
    fp.lemons_bought = 6 ∧ 
    fp.mangos_bought = 2 ∧ 
    fp.total_paid = 21 ∧ 
    papayas_bought fp 4 :=
sorry

end tom_buys_four_papayas_l1727_172799


namespace quadratic_vertex_l1727_172739

/-- The quadratic function f(x) = -2(x+3)^2 - 5 has vertex at (-3, -5) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -2 * (x + 3)^2 - 5
  (∀ x, f x ≤ f (-3)) ∧ f (-3) = -5 := by
  sorry

end quadratic_vertex_l1727_172739


namespace binary_110011_is_51_l1727_172744

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end binary_110011_is_51_l1727_172744


namespace truck_trailer_weights_l1727_172706

/-- Given the weights of trucks and trailers, prove their specific values -/
theorem truck_trailer_weights :
  ∀ (W_A W_B W_A' W_B' : ℝ),
    W_A + W_A' = 9000 →
    W_B + W_B' = 11000 →
    W_A' = 0.5 * W_A - 400 →
    W_B' = 0.4 * W_B + 500 →
    W_B = W_A + 2000 →
    W_A = 5500 ∧ W_B = 7500 ∧ W_A' = 2350 ∧ W_B' = 3500 := by
  sorry

end truck_trailer_weights_l1727_172706


namespace fixed_point_on_circle_l1727_172754

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix
def directrix : ℝ := -1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the circle
def circle_tangent_to_directrix (m : PointOnParabola) (p : ℝ × ℝ) : Prop :=
  let r := m.x - directrix
  (p.1 - m.x)^2 + (p.2 - m.y)^2 = r^2

-- Theorem statement
theorem fixed_point_on_circle (m : PointOnParabola) :
  circle_tangent_to_directrix m focus := by
  sorry

end fixed_point_on_circle_l1727_172754


namespace jack_payback_l1727_172775

/-- The amount borrowed by Jack -/
def principal : ℚ := 1200

/-- The interest rate as a decimal -/
def interestRate : ℚ := 1/10

/-- The amount Jack will pay back -/
def amountToPay : ℚ := principal * (1 + interestRate)

/-- Theorem stating that the amount Jack will pay back is 1320 -/
theorem jack_payback : amountToPay = 1320 := by sorry

end jack_payback_l1727_172775
