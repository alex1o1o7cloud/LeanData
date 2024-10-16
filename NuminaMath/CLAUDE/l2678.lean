import Mathlib

namespace NUMINAMATH_CALUDE_power_equality_n_equals_one_l2678_267897

theorem power_equality_n_equals_one :
  ∀ n : ℝ, (256 : ℝ) ^ (1/4 : ℝ) = 4 ^ n → n = 1 := by
sorry

end NUMINAMATH_CALUDE_power_equality_n_equals_one_l2678_267897


namespace NUMINAMATH_CALUDE_max_prime_value_l2678_267818

theorem max_prime_value (a b : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_eq : p = (b / 4) * Real.sqrt ((2 * a - b) / (2 * a + b))) : 
  p ≤ 5 ∧ ∃ (a' b' : ℕ), (5 : ℕ) = (b' / 4) * Real.sqrt ((2 * a' - b') / (2 * a' + b')) := by
  sorry

end NUMINAMATH_CALUDE_max_prime_value_l2678_267818


namespace NUMINAMATH_CALUDE_joydens_number_difference_l2678_267851

theorem joydens_number_difference (m j c : ℕ) : 
  m = j + 20 →
  j < c →
  c = 80 →
  m + j + c = 180 →
  c - j = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_joydens_number_difference_l2678_267851


namespace NUMINAMATH_CALUDE_integral_equals_four_l2678_267852

theorem integral_equals_four : ∫ x in (1:ℝ)..2, (3*x^2 - 2*x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_four_l2678_267852


namespace NUMINAMATH_CALUDE_vanessa_record_score_l2678_267857

/-- Vanessa's record-setting basketball score --/
theorem vanessa_record_score (total_team_score : ℕ) (other_players : ℕ) (other_players_avg : ℚ)
  (h1 : total_team_score = 48)
  (h2 : other_players = 6)
  (h3 : other_players_avg = 3.5) :
  total_team_score - (other_players : ℚ) * other_players_avg = 27 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_record_score_l2678_267857


namespace NUMINAMATH_CALUDE_petes_flag_shapes_petes_flag_total_shapes_l2678_267806

/-- Calculates the total number of shapes on Pete's flag based on US flag specifications -/
theorem petes_flag_shapes (us_stars : Nat) (us_stripes : Nat) : Nat :=
  let circles := us_stars / 2 - 3
  let squares := us_stripes * 2 + 6
  circles + squares

/-- Proves that the total number of shapes on Pete's flag is 54 -/
theorem petes_flag_total_shapes : 
  petes_flag_shapes 50 13 = 54 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_shapes_petes_flag_total_shapes_l2678_267806


namespace NUMINAMATH_CALUDE_percentage_of_b_l2678_267866

/-- Given that 12 is 6% of a, a certain percentage of b is 6, and c equals b / a,
    prove that the percentage of b is 6 / (200 * c) * 100 -/
theorem percentage_of_b (a b c : ℝ) (h1 : 0.06 * a = 12) (h2 : ∃ p, p * b = 6) (h3 : c = b / a) :
  ∃ p, p * b = 6 ∧ p * 100 = 6 / (200 * c) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_l2678_267866


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2678_267882

def z₁ : ℂ := Complex.I
def z₂ : ℂ := 1 + Complex.I

theorem z_in_second_quadrant : 
  let z : ℂ := z₁ * z₂
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2678_267882


namespace NUMINAMATH_CALUDE_chocolate_manufacturer_min_price_l2678_267853

/-- Calculates the minimum selling price per unit for a chocolate manufacturer --/
theorem chocolate_manufacturer_min_price
  (units : ℕ)
  (cost_per_unit : ℝ)
  (min_profit : ℝ)
  (h1 : units = 400)
  (h2 : cost_per_unit = 40)
  (h3 : min_profit = 40000) :
  (min_profit + units * cost_per_unit) / units = 140 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_manufacturer_min_price_l2678_267853


namespace NUMINAMATH_CALUDE_variance_2X_plus_1_l2678_267869

/-- A random variable following a Binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Variance of a Binomial distribution -/
def variance (X : BinomialDistribution) : ℝ :=
  X.n * X.p * (1 - X.p)

/-- Variance of a linear transformation of a random variable -/
def varianceLinearTransform (a b : ℝ) (X : BinomialDistribution) : ℝ :=
  a^2 * variance X

/-- Theorem: Variance of 2X+1 for X ~ B(10, 0.8) equals 6.4 -/
theorem variance_2X_plus_1 (X : BinomialDistribution) 
    (h2 : X.n = 10) (h3 : X.p = 0.8) : 
    varianceLinearTransform 2 1 X = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_variance_2X_plus_1_l2678_267869


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_l2678_267850

/-- Calculates the percentage decrease in salary after an initial increase -/
theorem salary_decrease_percentage 
  (original_salary : ℝ) 
  (initial_increase_percentage : ℝ) 
  (final_salary : ℝ) 
  (h1 : original_salary = 1000.0000000000001)
  (h2 : initial_increase_percentage = 10)
  (h3 : final_salary = 1045) :
  let increased_salary := original_salary * (1 + initial_increase_percentage / 100)
  let decrease_percentage := (1 - final_salary / increased_salary) * 100
  decrease_percentage = 5 := by
sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_l2678_267850


namespace NUMINAMATH_CALUDE_square_root_divided_by_thirteen_l2678_267817

theorem square_root_divided_by_thirteen : Real.sqrt 2704 / 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_thirteen_l2678_267817


namespace NUMINAMATH_CALUDE_age_difference_l2678_267808

theorem age_difference (A B : ℕ) : B = 41 → A + 10 = 2 * (B - 10) → A - B = 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2678_267808


namespace NUMINAMATH_CALUDE_permutations_of_five_l2678_267800

theorem permutations_of_five (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_five_l2678_267800


namespace NUMINAMATH_CALUDE_product_of_fractions_l2678_267875

/-- Prove that the product of 2/3 and 1 4/9 is equal to 26/27 -/
theorem product_of_fractions :
  (2 : ℚ) / 3 * (1 + 4 / 9) = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2678_267875


namespace NUMINAMATH_CALUDE_burning_time_3x5_rectangle_l2678_267845

/-- Represents a rectangular arrangement of toothpicks -/
structure ToothpickRectangle where
  rows : ℕ
  cols : ℕ
  total_toothpicks : ℕ

/-- Represents the burning properties of toothpicks -/
structure BurningProperties where
  single_toothpick_time : ℕ

/-- Calculates the maximum burning time for a toothpick rectangle -/
def max_burning_time (rect : ToothpickRectangle) (props : BurningProperties) : ℕ :=
  sorry

/-- Theorem: The maximum burning time for a 3x5 toothpick rectangle is 65 seconds -/
theorem burning_time_3x5_rectangle :
  let rect := ToothpickRectangle.mk 3 5 38
  let props := BurningProperties.mk 10
  max_burning_time rect props = 65 := by
  sorry

end NUMINAMATH_CALUDE_burning_time_3x5_rectangle_l2678_267845


namespace NUMINAMATH_CALUDE_hyperbola_and_angle_bisector_l2678_267827

-- Define the hyperbola Γ
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the line l
def line (x y : ℝ) : Prop :=
  x + y - 2 = 0

-- Define that l is parallel to one of the asymptotes and passes through a focus
def line_properties (a b : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), line x₀ y₀ ∧ 
  ((x₀ = a ∧ y₀ = 0) ∨ (x₀ = -a ∧ y₀ = 0)) ∧
  (∀ x y : ℝ, line x y → y = x ∨ y = -x)

-- Main theorem
theorem hyperbola_and_angle_bisector 
  (a b : ℝ) 
  (h : line_properties a b) :
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2 = 2) ∧
  (∃ P : ℝ × ℝ, 
    hyperbola a b P.1 P.2 ∧ 
    line P.1 P.2 ∧
    ∀ x y : ℝ, 3*x - y - 4 = 0 ↔ 
      (∃ t : ℝ, x = t*P.1 + (1-t)*(-2) ∧ y = t*P.2) ∨
      (∃ t : ℝ, x = t*P.1 + (1-t)*2 ∧ y = t*P.2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_angle_bisector_l2678_267827


namespace NUMINAMATH_CALUDE_sqrt3_times_3_minus_sqrt3_range_l2678_267896

theorem sqrt3_times_3_minus_sqrt3_range :
  2 < Real.sqrt 3 * (3 - Real.sqrt 3) ∧ Real.sqrt 3 * (3 - Real.sqrt 3) < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_times_3_minus_sqrt3_range_l2678_267896


namespace NUMINAMATH_CALUDE_street_length_proof_l2678_267807

/-- Proves that the length of a street is 1440 meters, given that a person crosses it in 12 minutes at a speed of 7.2 km per hour. -/
theorem street_length_proof (time : ℝ) (speed : ℝ) (length : ℝ) : 
  time = 12 →
  speed = 7.2 →
  length = speed * 1000 / 60 * time →
  length = 1440 := by
sorry

end NUMINAMATH_CALUDE_street_length_proof_l2678_267807


namespace NUMINAMATH_CALUDE_fifth_day_distance_l2678_267871

def running_sequence (n : ℕ) : ℕ := 2 + n - 1

theorem fifth_day_distance : running_sequence 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fifth_day_distance_l2678_267871


namespace NUMINAMATH_CALUDE_correct_proposition_l2678_267890

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 ≥ 0

-- Define proposition q
def q : Prop := 1 < 0

-- Theorem to prove
theorem correct_proposition : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l2678_267890


namespace NUMINAMATH_CALUDE_product_closure_l2678_267887

-- Define the set A
def A : Set ℤ := {z | ∃ (a b k n : ℤ), z = a^2 + k*a*b + n*b^2}

-- State the theorem
theorem product_closure (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := by
  sorry

end NUMINAMATH_CALUDE_product_closure_l2678_267887


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2678_267889

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) :
  z.im = -1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2678_267889


namespace NUMINAMATH_CALUDE_bluejay_league_members_l2678_267829

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 8

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := sock_cost / 2

/-- The total cost for one member's equipment (home and away sets) -/
def member_cost : ℕ := 2 * (sock_cost + tshirt_cost + cap_cost)

/-- The total expenditure for all members -/
def total_expenditure : ℕ := 3876

/-- The number of members in the Bluejay Basketball League -/
def num_members : ℕ := total_expenditure / member_cost

theorem bluejay_league_members : num_members = 84 := by
  sorry


end NUMINAMATH_CALUDE_bluejay_league_members_l2678_267829


namespace NUMINAMATH_CALUDE_hyperbola_proof_l2678_267868

def polar_equation (ρ φ : ℝ) : Prop := ρ = 36 / (4 - 5 * Real.cos φ)

theorem hyperbola_proof (ρ φ : ℝ) (h : polar_equation ρ φ) :
  ∃ (a b : ℝ), 
    (a = 16 ∧ b = 12) ∧ 
    (∃ (e : ℝ), e > 1 ∧ ρ = (e * (b^2 / a)) / (1 - e * Real.cos φ)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_proof_l2678_267868


namespace NUMINAMATH_CALUDE_divisor_problem_l2678_267894

theorem divisor_problem (n : ℕ+) : 
  (∃ k : ℕ, n = 2019 * k) →
  (∃ d : Fin 38 → ℕ+, 
    (∀ i j, i < j → d i < d j) ∧
    (∀ i, d i ∣ n) ∧
    (d 0 = 1) ∧
    (d 37 = n) ∧
    (n = d 18 * d 19)) →
  (n = 3^18 * 673 ∨ n = 673^18 * 3) := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l2678_267894


namespace NUMINAMATH_CALUDE_rectangular_to_polar_sqrt2_l2678_267826

theorem rectangular_to_polar_sqrt2 :
  ∃ (r : ℝ) (θ : ℝ), 
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r * Real.cos θ = Real.sqrt 2 ∧
    r * Real.sin θ = -Real.sqrt 2 ∧
    r = 2 ∧
    θ = 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_sqrt2_l2678_267826


namespace NUMINAMATH_CALUDE_height_weight_relationship_l2678_267804

/-- Represents the coefficient of determination (R²) in a linear regression model -/
def R_squared : ℝ := 0.64

/-- The proportion of variation explained by the model -/
def variation_explained : ℝ := R_squared

/-- The proportion of variation not explained by the model (random error) -/
def variation_unexplained : ℝ := 1 - R_squared

theorem height_weight_relationship :
  variation_explained = 0.64 ∧
  variation_unexplained = 0.36 ∧
  variation_explained + variation_unexplained = 1 := by
  sorry

#eval R_squared
#eval variation_explained
#eval variation_unexplained

end NUMINAMATH_CALUDE_height_weight_relationship_l2678_267804


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2678_267865

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Existence of the hyperbola
  (∃ x : ℝ, x = 2 ∧ x^2 / a^2 - 0^2 / b^2 = 1) →  -- Right vertex at (2, 0)
  (∃ c : ℝ, c / a = 3/2) →  -- Eccentricity is 3/2
  (∃ x y : ℝ, y^2 = 8*x) →  -- Existence of the parabola
  (∀ x y : ℝ, x^2 / 4 - y^2 / 5 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2678_267865


namespace NUMINAMATH_CALUDE_jane_dolls_l2678_267892

theorem jane_dolls (total : ℕ) (difference : ℕ) : total = 32 → difference = 6 → ∃ jane : ℕ, jane = 13 ∧ jane + (jane + difference) = total := by
  sorry

end NUMINAMATH_CALUDE_jane_dolls_l2678_267892


namespace NUMINAMATH_CALUDE_smallest_group_size_fifty_nine_satisfies_conditions_fewest_students_l2678_267812

theorem smallest_group_size (N : ℕ) : 
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) → N ≥ 59 :=
by sorry

theorem fifty_nine_satisfies_conditions : 
  (59 % 5 = 2) ∧ (59 % 6 = 3) ∧ (59 % 8 = 4) :=
by sorry

theorem fewest_students : 
  ∃ (N : ℕ), (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) ∧ 
  (∀ (M : ℕ), (M % 5 = 2) ∧ (M % 6 = 3) ∧ (M % 8 = 4) → M ≥ N) ∧
  N = 59 :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_fifty_nine_satisfies_conditions_fewest_students_l2678_267812


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2678_267873

/-- Represents a seating arrangement -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Represents a married couple -/
structure Couple := (husband : Fin 6) (wife : Fin 6)

/-- Represents a profession -/
def Profession := Fin 3

/-- Check if two positions are adjacent or opposite on a 12-seat round table -/
def isAdjacentOrOpposite (a b : Fin 12) : Prop := 
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a + 6 = b) ∨ (b + 6 = a)

/-- Check if a seating arrangement is valid -/
def isValidArrangement (s : SeatingArrangement) (couples : Fin 6 → Couple) (professions : Fin 12 → Profession) : Prop :=
  ∀ i j : Fin 12, 
    -- Men and women alternate
    (i.val % 2 = 0 ↔ j.val % 2 = 1) →
    -- No one sits next to or across from their spouse
    (∃ c : Fin 6, (couples c).husband = s i ∧ (couples c).wife = s j) →
    ¬ isAdjacentOrOpposite i j ∧
    -- No one sits next to someone of the same profession
    (isAdjacentOrOpposite i j → professions (s i) ≠ professions (s j))

/-- The main theorem stating the number of valid seating arrangements -/
theorem seating_arrangements_count :
  ∃ (arrangements : Finset SeatingArrangement) (couples : Fin 6 → Couple) (professions : Fin 12 → Profession),
    arrangements.card = 2880 ∧
    ∀ s ∈ arrangements, isValidArrangement s couples professions :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2678_267873


namespace NUMINAMATH_CALUDE_foreign_language_books_l2678_267810

theorem foreign_language_books (total : ℝ) 
  (h1 : total * (36 / 100) = total - (total * (27 / 100) + 185))
  (h2 : total * (27 / 100) = total * (36 / 100) * (75 / 100))
  (h3 : 185 = total - (total * (36 / 100) + total * (27 / 100))) :
  total = 500 := by sorry

end NUMINAMATH_CALUDE_foreign_language_books_l2678_267810


namespace NUMINAMATH_CALUDE_odd_roll_probability_theorem_l2678_267816

/-- Represents a standard six-sided die -/
inductive Die : Type
| one | two | three | four | five | six

/-- Represents a sequence of die rolls -/
def RollSequence := List Die

/-- Checks if a sequence ends with 1-2-3 -/
def endsWithOneTwoThree (seq : RollSequence) : Prop :=
  seq.reverse.take 3 = [Die.three, Die.two, Die.one]

/-- The probability of rolling an odd number of times before getting 1-2-3 -/
def oddRollProbability : ℚ :=
  216 / 431

theorem odd_roll_probability_theorem :
  oddRollProbability = 216 / 431 :=
sorry

end NUMINAMATH_CALUDE_odd_roll_probability_theorem_l2678_267816


namespace NUMINAMATH_CALUDE_point_distance_on_curve_l2678_267858

theorem point_distance_on_curve (e c d : ℝ) : 
  e > 0 →
  c ≠ d →
  c^2 + (Real.sqrt e)^6 = 3 * (Real.sqrt e)^3 * c + 1 →
  d^2 + (Real.sqrt e)^6 = 3 * (Real.sqrt e)^3 * d + 1 →
  |c - d| = |Real.sqrt (5 * e^3 + 4)| :=
by sorry

end NUMINAMATH_CALUDE_point_distance_on_curve_l2678_267858


namespace NUMINAMATH_CALUDE_samara_alligators_l2678_267840

theorem samara_alligators (group_size : ℕ) (friends_count : ℕ) (friends_average : ℕ) (total_alligators : ℕ) :
  group_size = friends_count + 1 →
  friends_count = 3 →
  friends_average = 10 →
  total_alligators = 50 →
  total_alligators = friends_count * friends_average + (total_alligators - friends_count * friends_average) →
  (total_alligators - friends_count * friends_average) = 20 := by
  sorry

end NUMINAMATH_CALUDE_samara_alligators_l2678_267840


namespace NUMINAMATH_CALUDE_total_players_l2678_267834

theorem total_players (kabaddi : ℕ) (kho_kho_only : ℕ) (both : ℕ) 
  (h1 : kabaddi = 10) 
  (h2 : kho_kho_only = 40) 
  (h3 : both = 5) : 
  kabaddi + kho_kho_only - both = 50 := by
  sorry

#check total_players

end NUMINAMATH_CALUDE_total_players_l2678_267834


namespace NUMINAMATH_CALUDE_data_mode_and_mean_l2678_267844

def data : List ℕ := [5, 6, 8, 6, 8, 8, 8]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem data_mode_and_mean :
  mode data = 8 ∧ mean data = 7 := by
  sorry

end NUMINAMATH_CALUDE_data_mode_and_mean_l2678_267844


namespace NUMINAMATH_CALUDE_spanish_only_count_l2678_267886

/-- Represents the number of students in different language classes -/
structure LanguageClasses where
  total : ℕ
  french : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of students taking Spanish only -/
def spanishOnly (lc : LanguageClasses) : ℕ :=
  lc.total - lc.french - lc.neither + lc.both

/-- Theorem stating the number of students taking Spanish only -/
theorem spanish_only_count (lc : LanguageClasses) 
  (h1 : lc.total = 28)
  (h2 : lc.french = 5)
  (h3 : lc.both = 4)
  (h4 : lc.neither = 13) :
  spanishOnly lc = 10 := by
  sorry

#check spanish_only_count

end NUMINAMATH_CALUDE_spanish_only_count_l2678_267886


namespace NUMINAMATH_CALUDE_digit_add_sequence_contains_even_l2678_267837

/-- A sequence of natural numbers where each term is obtained from the previous term
    by adding one of its nonzero digits. -/
def DigitAddSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ d : ℕ, d > 0 ∧ d < 10 ∧ d ∣ a n ∧ a (n + 1) = a n + d

/-- The theorem stating that a DigitAddSequence contains an even number. -/
theorem digit_add_sequence_contains_even (a : ℕ → ℕ) (h : DigitAddSequence a) :
  ∃ n : ℕ, Even (a n) :=
sorry

end NUMINAMATH_CALUDE_digit_add_sequence_contains_even_l2678_267837


namespace NUMINAMATH_CALUDE_triangle_median_theorem_l2678_267864

-- Define the triangle and its medians
structure Triangle :=
  (D E F : ℝ × ℝ)
  (DP EQ : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  -- DP and EQ are medians
  ∃ P Q : ℝ × ℝ,
    t.DP = P - t.D ∧
    t.EQ = Q - t.E ∧
    P = (t.E + t.F) / 2 ∧
    Q = (t.D + t.F) / 2 ∧
  -- DP and EQ are perpendicular
  t.DP.1 * t.EQ.1 + t.DP.2 * t.EQ.2 = 0 ∧
  -- Lengths of DP and EQ
  Real.sqrt (t.DP.1^2 + t.DP.2^2) = 18 ∧
  Real.sqrt (t.EQ.1^2 + t.EQ.2^2) = 24

-- Theorem statement
theorem triangle_median_theorem (t : Triangle) (h : is_valid_triangle t) :
  Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2) = 8 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_triangle_median_theorem_l2678_267864


namespace NUMINAMATH_CALUDE_max_rectangle_area_l2678_267824

/-- The maximum area of a rectangle given constraints --/
theorem max_rectangle_area (perimeter : ℝ) (min_length min_width : ℝ) :
  perimeter = 400 ∧ min_length = 100 ∧ min_width = 50 →
  ∃ (length width : ℝ),
    length ≥ min_length ∧
    width ≥ min_width ∧
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℝ),
      l ≥ min_length →
      w ≥ min_width →
      2 * (l + w) = perimeter →
      l * w ≤ length * width ∧
      length * width = 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l2678_267824


namespace NUMINAMATH_CALUDE_average_weight_problem_l2678_267843

theorem average_weight_problem (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2678_267843


namespace NUMINAMATH_CALUDE_x_value_proof_l2678_267856

theorem x_value_proof (x : ℝ) : (5 * x - 3)^3 = Real.sqrt 64 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2678_267856


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l2678_267820

def problem (workers : ℕ) (supervisors : ℕ) (initial_avg : ℝ) (supervisor_a : ℝ) (supervisor_b : ℝ) (supervisor_c : ℝ) (new_avg : ℝ) : Prop :=
  let total_people := workers + supervisors
  let initial_total := initial_avg * total_people
  let workers_supervisors_ab_total := initial_total - supervisor_c
  let new_total := new_avg * total_people
  let salary_difference := initial_total - new_total
  let new_supervisor_salary := supervisor_c - salary_difference
  new_supervisor_salary = 4600

theorem new_supervisor_salary :
  problem 15 3 5300 6200 7200 8200 5100 :=
by
  sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l2678_267820


namespace NUMINAMATH_CALUDE_mary_vacuum_charges_l2678_267836

/-- The number of times Mary needs to charge her vacuum cleaner to clean her whole house -/
def charges_needed (battery_duration : ℕ) (time_per_room : ℕ) (num_bedrooms : ℕ) (num_kitchen : ℕ) (num_living_room : ℕ) : ℕ :=
  let total_rooms := num_bedrooms + num_kitchen + num_living_room
  let total_time := time_per_room * total_rooms
  (total_time + battery_duration - 1) / battery_duration

theorem mary_vacuum_charges :
  charges_needed 10 4 3 1 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_vacuum_charges_l2678_267836


namespace NUMINAMATH_CALUDE_shaggy_seed_count_l2678_267867

/-- Represents the number of seeds Shaggy ate -/
def shaggy_seeds : ℕ := 54

/-- Represents the total number of seeds -/
def total_seeds : ℕ := 60

/-- Represents the ratio of Shaggy's berry eating speed to Fluffball's -/
def berry_speed_ratio : ℕ := 6

/-- Represents the ratio of Shaggy's seed eating speed to Fluffball's -/
def seed_speed_ratio : ℕ := 3

/-- Represents the ratio of berries Shaggy ate to Fluffball -/
def berry_ratio : ℕ := 2

theorem shaggy_seed_count : 
  50 < total_seeds ∧ 
  total_seeds < 65 ∧ 
  berry_speed_ratio = 6 ∧ 
  seed_speed_ratio = 3 ∧ 
  berry_ratio = 2 → 
  shaggy_seeds = 54 := by sorry

end NUMINAMATH_CALUDE_shaggy_seed_count_l2678_267867


namespace NUMINAMATH_CALUDE_tiling_impossible_l2678_267874

/-- Represents a 1 × 3 strip used for tiling -/
structure Strip :=
  (length : Nat)
  (width : Nat)
  (h_length : length = 3)
  (h_width : width = 1)

/-- Represents the figure to be tiled -/
structure Figure :=
  (total_squares : Nat)
  (color1_squares : Nat)
  (color2_squares : Nat)
  (h_total : total_squares = color1_squares + color2_squares)
  (h_color1 : color1_squares = 7)
  (h_color2 : color2_squares = 8)

/-- Represents a tiling of the figure with strips -/
structure Tiling :=
  (figure : Figure)
  (strips : List Strip)
  (h_cover : ∀ s ∈ strips, s.length = 3 ∧ s.width = 1)
  (h_no_overlap : List.Nodup strips)
  (h_complete : strips.length * 3 = figure.total_squares)

/-- The main theorem stating that tiling is impossible -/
theorem tiling_impossible (f : Figure) : ¬ ∃ t : Tiling, t.figure = f := by
  sorry

end NUMINAMATH_CALUDE_tiling_impossible_l2678_267874


namespace NUMINAMATH_CALUDE_pizza_problem_l2678_267878

/-- The number of triple cheese pizzas purchased -/
def T : ℕ := 10

/-- The number of meat lovers pizzas purchased -/
def M : ℕ := 9

/-- The standard price of a pizza in dollars -/
def standard_price : ℕ := 5

/-- The total cost in dollars -/
def total_cost : ℕ := 55

/-- The cost of triple cheese pizzas under the special pricing -/
def triple_cheese_cost (n : ℕ) : ℕ := (n / 2) * standard_price

/-- The cost of meat lovers pizzas under the special pricing -/
def meat_lovers_cost (n : ℕ) : ℕ := ((n / 3) * 2) * standard_price

theorem pizza_problem : 
  triple_cheese_cost T + meat_lovers_cost M = total_cost :=
sorry

end NUMINAMATH_CALUDE_pizza_problem_l2678_267878


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2678_267809

theorem ceiling_floor_sum : ⌈(7:ℝ)/3⌉ + ⌊-(7:ℝ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2678_267809


namespace NUMINAMATH_CALUDE_soccer_balls_with_holes_l2678_267813

theorem soccer_balls_with_holes (total_soccer : ℕ) (total_basketball : ℕ) (basketball_with_holes : ℕ) (total_without_holes : ℕ) :
  total_soccer = 40 →
  total_basketball = 15 →
  basketball_with_holes = 7 →
  total_without_holes = 18 →
  total_soccer - (total_without_holes - (total_basketball - basketball_with_holes)) = 30 := by
sorry

end NUMINAMATH_CALUDE_soccer_balls_with_holes_l2678_267813


namespace NUMINAMATH_CALUDE_cindy_calculation_l2678_267891

theorem cindy_calculation (x : ℝ) (h : (x + 7) * 5 = 260) : 5 * x + 7 = 232 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l2678_267891


namespace NUMINAMATH_CALUDE_train_crossing_time_l2678_267838

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 48 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2678_267838


namespace NUMINAMATH_CALUDE_function_odd_and_periodic_l2678_267803

/-- A function f: ℝ → ℝ satisfying f(10+x) = f(10-x) and f(20-x) = -f(20+x) for all x ∈ ℝ is odd and has period 20. -/
theorem function_odd_and_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x, f (10 + x) = f (10 - x))
  (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 20) = f x) :=
sorry

end NUMINAMATH_CALUDE_function_odd_and_periodic_l2678_267803


namespace NUMINAMATH_CALUDE_mapping_has_output_l2678_267839

-- Define sets M and N
variable (M N : Type)

-- Define the mapping f from M to N
variable (f : M → N)

-- Theorem statement
theorem mapping_has_output : ∀ (x : M), ∃ (y : N), f x = y := by
  sorry

end NUMINAMATH_CALUDE_mapping_has_output_l2678_267839


namespace NUMINAMATH_CALUDE_circle_angle_sum_l2678_267815

/-- Given a circle divided into 12 equal arcs, this theorem proves that the sum of
    half the central angle spanning 2 arcs and half the central angle spanning 4 arcs
    is equal to 90 degrees. -/
theorem circle_angle_sum (α β : Real) : 
  (∀ (n : Nat), n ≤ 12 → 360 / 12 * n = 30 * n) →
  α = (2 * 360 / 12) / 2 →
  β = (4 * 360 / 12) / 2 →
  α + β = 90 := by
sorry

end NUMINAMATH_CALUDE_circle_angle_sum_l2678_267815


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2678_267831

/-- A line y = mx + (2m+1) always passes through the point (-2, 1) for any real m -/
theorem line_passes_through_fixed_point (m : ℝ) : 
  let f : ℝ → ℝ := fun x => m * x + (2 * m + 1)
  f (-2) = 1 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2678_267831


namespace NUMINAMATH_CALUDE_machine_job_time_l2678_267821

theorem machine_job_time (y : ℝ) : 
  (1 / (y + 8) + 1 / (y + 3) + 1 / (1.5 * y) = 1 / y) →
  y = (-25 + Real.sqrt 421) / 6 :=
by sorry

end NUMINAMATH_CALUDE_machine_job_time_l2678_267821


namespace NUMINAMATH_CALUDE_sum_P_equals_97335_l2678_267832

/-- P(n) represents the product of all non-zero digits of a positive integer n -/
def P (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n < 10 then n
  else let d := n % 10
       let r := n / 10
       if d = 0 then P r
       else d * P r

/-- The sum of P(n) for n from 1 to 999 -/
def sum_P : ℕ := (List.range 999).map (fun i => P (i + 1)) |>.sum

theorem sum_P_equals_97335 : sum_P = 97335 := by
  sorry

end NUMINAMATH_CALUDE_sum_P_equals_97335_l2678_267832


namespace NUMINAMATH_CALUDE_special_circle_equation_l2678_267893

/-- A circle with center on the y-axis passing through (3, 1) and tangent to x-axis -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_y_axis : center.1 = 0
  passes_through_point : (3 - center.1)^2 + (1 - center.2)^2 = radius^2
  tangent_to_x_axis : center.2 = radius

/-- The equation of the special circle is x^2 + y^2 - 10y = 0 -/
theorem special_circle_equation (c : SpecialCircle) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔ x^2 + y^2 - 10*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_special_circle_equation_l2678_267893


namespace NUMINAMATH_CALUDE_boy_age_problem_l2678_267861

theorem boy_age_problem (total_boys : ℕ) (avg_all : ℕ) (avg_first : ℕ) (avg_last : ℕ) 
  (h_total : total_boys = 11)
  (h_avg_all : avg_all = 50)
  (h_avg_first : avg_first = 49)
  (h_avg_last : avg_last = 52) :
  (total_boys * avg_all : ℕ) = 
  (6 * avg_first : ℕ) + (6 * avg_last : ℕ) - 56 := by
  sorry

#check boy_age_problem

end NUMINAMATH_CALUDE_boy_age_problem_l2678_267861


namespace NUMINAMATH_CALUDE_sum_of_products_equals_25079720_l2678_267863

def T : Finset ℕ := Finset.image (fun i => 3^i) (Finset.range 8)

def M : ℕ := (Finset.sum T fun x => 
  (Finset.sum (T.erase x) fun y => x * y))

theorem sum_of_products_equals_25079720 : M = 25079720 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_equals_25079720_l2678_267863


namespace NUMINAMATH_CALUDE_square_of_geometric_is_geometric_l2678_267835

-- Define a geometric sequence
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Statement to prove
theorem square_of_geometric_is_geometric (a : ℕ → ℝ) (h : IsGeometric a) :
  IsGeometric (fun n ↦ (a n)^2) := by
  sorry

end NUMINAMATH_CALUDE_square_of_geometric_is_geometric_l2678_267835


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2678_267823

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → x^3 + x ≥ 0) ↔ 
  (∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ x^3 + x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2678_267823


namespace NUMINAMATH_CALUDE_expand_expression_l2678_267833

theorem expand_expression (x : ℝ) : 5 * (x + 3) * (2 * x - 4) = 10 * x^2 + 10 * x - 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2678_267833


namespace NUMINAMATH_CALUDE_find_t_value_l2678_267862

theorem find_t_value (t : ℝ) : 
  let A : Set ℝ := {-4, t^2}
  let B : Set ℝ := {t-5, 9, 1-t}
  9 ∈ A ∩ B → t = -3 :=
by sorry

end NUMINAMATH_CALUDE_find_t_value_l2678_267862


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l2678_267825

/-- Calculates the correct average marks after correcting an error in one student's mark -/
theorem correct_average_after_error_correction 
  (num_students : ℕ) 
  (initial_average : ℚ) 
  (wrong_mark : ℚ) 
  (correct_mark : ℚ) : 
  num_students = 10 → 
  initial_average = 100 → 
  wrong_mark = 60 → 
  correct_mark = 10 → 
  (initial_average * num_students - wrong_mark + correct_mark) / num_students = 95 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l2678_267825


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2678_267895

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin (α + π/4) = Real.sqrt 3 / 3) : 
  Real.cos α = (-2 * Real.sqrt 3 + Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2678_267895


namespace NUMINAMATH_CALUDE_sum_mod_nine_l2678_267830

theorem sum_mod_nine :
  (2 + 333 + 5555 + 77777 + 999999 + 2222222 + 44444444 + 666666666) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l2678_267830


namespace NUMINAMATH_CALUDE_sarah_apples_l2678_267884

theorem sarah_apples (boxes : ℕ) (apples_per_box : ℕ) (h1 : boxes = 7) (h2 : apples_per_box = 7) :
  boxes * apples_per_box = 49 := by
  sorry

end NUMINAMATH_CALUDE_sarah_apples_l2678_267884


namespace NUMINAMATH_CALUDE_propositions_true_l2678_267822

theorem propositions_true (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 - b^2 = 1 → a - b < 1) ∧
  (Real.exp a - Real.exp b = 1 → a - b < 1) := by
  sorry

end NUMINAMATH_CALUDE_propositions_true_l2678_267822


namespace NUMINAMATH_CALUDE_pascals_triangle_51st_row_third_number_l2678_267876

theorem pascals_triangle_51st_row_third_number : 
  (Nat.choose 51 2) = 1275 := by sorry

end NUMINAMATH_CALUDE_pascals_triangle_51st_row_third_number_l2678_267876


namespace NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l2678_267819

theorem percentage_of_red_non_honda_cars
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (honda_red_ratio : ℚ)
  (total_red_ratio : ℚ)
  (h1 : total_cars = 9000)
  (h2 : honda_cars = 5000)
  (h3 : honda_red_ratio = 90 / 100)
  (h4 : total_red_ratio = 60 / 100)
  : (((total_red_ratio * total_cars) - (honda_red_ratio * honda_cars)) /
     (total_cars - honda_cars)) = 225 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l2678_267819


namespace NUMINAMATH_CALUDE_polynomial_divisibility_condition_l2678_267848

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Definition of divisibility for integers -/
def divides (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

/-- Definition of an odd prime number -/
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

/-- The main theorem -/
theorem polynomial_divisibility_condition (f : IntPolynomial) :
  (∀ p : ℕ, is_odd_prime p → divides (f.eval p) ((p - 3).factorial + (p + 1) / 2)) →
  (f = Polynomial.X) ∨ (f = -Polynomial.X) ∨ (f = Polynomial.C 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_condition_l2678_267848


namespace NUMINAMATH_CALUDE_coin_difference_l2678_267860

def coin_values : List Nat := [5, 10, 25, 50]

def total_amount : Nat := 60

def min_coins (values : List Nat) (amount : Nat) : Nat :=
  sorry

def max_coins (values : List Nat) (amount : Nat) : Nat :=
  sorry

theorem coin_difference :
  max_coins coin_values total_amount - min_coins coin_values total_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_coin_difference_l2678_267860


namespace NUMINAMATH_CALUDE_work_completion_time_l2678_267888

theorem work_completion_time (a b : ℝ) (h1 : b = 20) 
  (h2 : 4 * (1/a + 1/b) = 0.4666666666666667) : a = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2678_267888


namespace NUMINAMATH_CALUDE_first_month_sale_l2678_267802

def sale_month2 : ℕ := 8927
def sale_month3 : ℕ := 8855
def sale_month4 : ℕ := 9230
def sale_month5 : ℕ := 8562
def sale_month6 : ℕ := 6991
def average_sale : ℕ := 8500
def num_months : ℕ := 6

theorem first_month_sale :
  sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6 +
  (average_sale * num_months - (sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6)) =
  average_sale * num_months :=
by sorry

end NUMINAMATH_CALUDE_first_month_sale_l2678_267802


namespace NUMINAMATH_CALUDE_kangaroo_jump_distance_l2678_267885

/-- Proves that the kangaroo's jump distance is 35 inches given the conditions of the jumping contest. -/
theorem kangaroo_jump_distance (G F M K : ℕ) : 
  G = F + 19 →
  M = F - 12 →
  K = 2 * F - 5 →
  G = 39 →
  K = 35 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_jump_distance_l2678_267885


namespace NUMINAMATH_CALUDE_problem_solution_l2678_267883

def A (a : ℝ) : Set ℝ := {x | 2 * a - 3 < x ∧ x < a + 1}

def B : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem problem_solution :
  (∀ x : ℝ, x ∈ A 0 ∩ B ↔ -1 < x ∧ x < 1) ∧
  (∀ a : ℝ, A a ∩ (Set.univ \ B) = A a ↔ a ≤ -2 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2678_267883


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2678_267805

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 - Complex.I) = 3 + Complex.I) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2678_267805


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2678_267811

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b * Real.cos A = (Real.sqrt 2 * c - a) * Real.cos B →
  B = π / 4 →
  C > π / 2 →
  a = 4 →
  b = 3 →
  (1 / 2) * a * b * Real.sin C = 4 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2678_267811


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2678_267859

theorem cubic_roots_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  (a / ((1/a) + b*c)) + (b / ((1/b) + c*a)) + (c / ((1/c) + a*b)) = 175/11 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2678_267859


namespace NUMINAMATH_CALUDE_damien_jogging_days_l2678_267879

/-- Represents the number of miles Damien jogs per day -/
def miles_per_day : ℕ := 5

/-- Represents the total number of miles Damien jogs over three weeks -/
def total_miles : ℕ := 75

/-- Calculates the number of days Damien jogs over three weeks -/
def days_jogged : ℕ := total_miles / miles_per_day

theorem damien_jogging_days :
  days_jogged = 15 := by sorry

end NUMINAMATH_CALUDE_damien_jogging_days_l2678_267879


namespace NUMINAMATH_CALUDE_part_one_part_two_l2678_267828

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.A + t.c * Real.sin t.C - Real.sqrt 2 * t.a * Real.sin t.C = t.b * Real.sin t.B

-- Theorem for part (I)
theorem part_one (t : Triangle) (h : given_condition t) : t.B = Real.pi / 4 := by
  sorry

-- Theorem for part (II)
theorem part_two (t : Triangle) (h1 : given_condition t) (h2 : t.A = 5 * Real.pi / 12) (h3 : t.b = 2) :
  t.a = 1 + Real.sqrt 3 ∧ t.c = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2678_267828


namespace NUMINAMATH_CALUDE_area_minimized_at_k_equals_one_l2678_267854

/-- Represents a planar region defined by a system of inequalities -/
def PlanarRegion := Set (ℝ × ℝ)

/-- Computes the area of a planar region -/
noncomputable def area (Ω : PlanarRegion) : ℝ := sorry

/-- The system of inequalities that defines Ω -/
def systemOfInequalities (k : ℝ) : PlanarRegion := sorry

theorem area_minimized_at_k_equals_one (k : ℝ) (hk : k ≥ 0) :
  let Ω := systemOfInequalities k
  ∀ k' ≥ 0, area Ω ≤ area (systemOfInequalities k') → k = 1 :=
sorry

end NUMINAMATH_CALUDE_area_minimized_at_k_equals_one_l2678_267854


namespace NUMINAMATH_CALUDE_small_sphere_acceleration_l2678_267842

/-- The acceleration of a small charged sphere after material removal from a larger charged sphere -/
theorem small_sphere_acceleration
  (k : ℝ) -- Coulomb's constant
  (q Q : ℝ) -- Charges of small and large spheres
  (r R : ℝ) -- Radii of small and large spheres
  (m : ℝ) -- Mass of small sphere
  (L S : ℝ) -- Distances
  (g : ℝ) -- Acceleration due to gravity
  (h_r_small : r < R)
  (h_initial_balance : k * q * Q / (L + R)^2 = m * g)
  : ∃ (a : ℝ), a = (k * q * Q * r^3) / (m * R^3 * (L + 2*R - S)^2) :=
sorry

end NUMINAMATH_CALUDE_small_sphere_acceleration_l2678_267842


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2678_267855

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let original_area := L * W
  let new_length := L * 1.2
  let new_width := W * 1.2
  let new_area := new_length * new_width
  (new_area - original_area) / original_area * 100 = 44 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2678_267855


namespace NUMINAMATH_CALUDE_cooking_participants_l2678_267880

/-- The number of people in a curriculum group with various activities -/
structure CurriculumGroup where
  yoga : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- The total number of people studying cooking in the curriculum group -/
def totalCooking (g : CurriculumGroup) : ℕ :=
  g.cookingOnly + (g.cookingAndYoga - g.allCurriculums) + 
  (g.cookingAndWeaving - g.allCurriculums) + g.allCurriculums

/-- Theorem stating that the number of people studying cooking is 9 -/
theorem cooking_participants (g : CurriculumGroup) 
  (h1 : g.yoga = 25)
  (h2 : g.weaving = 8)
  (h3 : g.cookingOnly = 2)
  (h4 : g.cookingAndYoga = 7)
  (h5 : g.allCurriculums = 3)
  (h6 : g.cookingAndWeaving = 3) :
  totalCooking g = 9 := by
  sorry

end NUMINAMATH_CALUDE_cooking_participants_l2678_267880


namespace NUMINAMATH_CALUDE_closest_point_l2678_267898

def u (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1 + 5*s
  | 1 => -2 + 3*s
  | 2 => -4 - 2*s

def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3
  | 1 => 3
  | 2 => 4

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 3
  | 2 => -2

theorem closest_point (s : ℝ) : 
  (∀ t : ℝ, ‖u s - b‖^2 ≤ ‖u t - b‖^2) ↔ s = 9/38 := by sorry

end NUMINAMATH_CALUDE_closest_point_l2678_267898


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2678_267870

/-- Two 2D vectors are parallel if the cross product of their coordinates is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (m, m + 1)
  parallel a b → m = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2678_267870


namespace NUMINAMATH_CALUDE_roots_sequence_sum_l2678_267872

theorem roots_sequence_sum (p q a b : ℝ) : 
  p > 0 → 
  q > 0 → 
  a ≠ b →
  a^2 - p*a + q = 0 →
  b^2 - p*b + q = 0 →
  (∃ d : ℝ, (a = -4 + d ∧ b = -4 + 2*d) ∨ (b = -4 + d ∧ a = -4 + 2*d)) →
  (∃ r : ℝ, (a = -4*r ∧ b = -4*r^2) ∨ (b = -4*r ∧ a = -4*r^2)) →
  p + q = 26 := by
sorry

end NUMINAMATH_CALUDE_roots_sequence_sum_l2678_267872


namespace NUMINAMATH_CALUDE_dandelion_seed_production_l2678_267841

-- Define the number of seeds produced by a single dandelion plant
def seeds_per_plant : ℕ := 50

-- Define the germination rate (half of the seeds)
def germination_rate : ℚ := 1 / 2

-- Theorem statement
theorem dandelion_seed_production :
  let initial_seeds := seeds_per_plant
  let germinated_plants := (initial_seeds : ℚ) * germination_rate
  let total_seeds := (germinated_plants * seeds_per_plant : ℚ)
  total_seeds = 1250 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seed_production_l2678_267841


namespace NUMINAMATH_CALUDE_initial_bench_press_weight_l2678_267849

/-- The initial bench press weight before injury -/
def W : ℝ := 500

/-- The bench press weight after injury -/
def after_injury : ℝ := 0.2 * W

/-- The bench press weight after training -/
def after_training : ℝ := 3 * after_injury

/-- The final bench press weight -/
def final_weight : ℝ := 300

theorem initial_bench_press_weight :
  W = 500 ∧ after_injury = 0.2 * W ∧ after_training = 3 * after_injury ∧ after_training = final_weight := by
  sorry

end NUMINAMATH_CALUDE_initial_bench_press_weight_l2678_267849


namespace NUMINAMATH_CALUDE_weight_distribution_problem_l2678_267881

theorem weight_distribution_problem :
  ∃! (a b c : ℕ), a + b + c = 100 ∧ a + 10 * b + 50 * c = 500 ∧ (a, b, c) = (60, 39, 1) := by
  sorry

end NUMINAMATH_CALUDE_weight_distribution_problem_l2678_267881


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2678_267814

theorem arctan_equation_solution :
  ∀ x : ℝ, Real.arctan (1 / x) + Real.arctan (1 / x^2) = π / 4 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2678_267814


namespace NUMINAMATH_CALUDE_octagon_area_l2678_267801

-- Define the octagon's vertices
def octagon_vertices : List (ℝ × ℝ) :=
  [(0, 0), (1, 3), (2.5, 4), (4.5, 4), (6, 1), (4.5, -2), (2.5, -3), (1, -3)]

-- Define the function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ :=
  sorry

-- Theorem statement
theorem octagon_area :
  polygon_area octagon_vertices = 34 :=
sorry

end NUMINAMATH_CALUDE_octagon_area_l2678_267801


namespace NUMINAMATH_CALUDE_first_protest_duration_l2678_267846

/-- 
Given a person who attends two protests where the second protest duration is 25% longer 
than the first, and the total time spent protesting is 9 days, prove that the duration 
of the first protest is 4 days.
-/
theorem first_protest_duration (first_duration : ℝ) 
  (h1 : first_duration > 0)
  (h2 : first_duration + (1.25 * first_duration) = 9) : 
  first_duration = 4 := by
sorry

end NUMINAMATH_CALUDE_first_protest_duration_l2678_267846


namespace NUMINAMATH_CALUDE_unique_valid_sequence_l2678_267847

/-- Represents a sequence of 5 missile numbers. -/
def MissileSequence := Fin 5 → Nat

/-- The total number of missiles. -/
def totalMissiles : Nat := 50

/-- Checks if a sequence is valid according to the problem conditions. -/
def isValidSequence (seq : MissileSequence) : Prop :=
  ∀ i j : Fin 5, i < j →
    (seq i < seq j) ∧
    (seq j ≤ totalMissiles) ∧
    (∃ k : Nat, seq j - seq i = k * (j - i))

/-- The specific sequence given in the correct answer. -/
def correctSequence : MissileSequence :=
  fun i => [3, 13, 23, 33, 43].get i

/-- Theorem stating that the correct sequence is the only valid sequence. -/
theorem unique_valid_sequence :
  (isValidSequence correctSequence) ∧
  (∀ seq : MissileSequence, isValidSequence seq → seq = correctSequence) := by
  sorry


end NUMINAMATH_CALUDE_unique_valid_sequence_l2678_267847


namespace NUMINAMATH_CALUDE_flour_sack_cost_l2678_267877

/-- Represents the cost and customs scenario for flour sacks --/
structure FlourScenario where
  sack_cost : ℕ  -- Cost of one sack of flour in pesetas
  customs_duty : ℕ  -- Customs duty per sack in pesetas
  truck1_sacks : ℕ := 118  -- Number of sacks in first truck
  truck2_sacks : ℕ := 40   -- Number of sacks in second truck
  truck1_left : ℕ := 10    -- Sacks left by first truck
  truck2_left : ℕ := 4     -- Sacks left by second truck
  truck1_pay : ℕ := 800    -- Additional payment by first truck
  truck2_receive : ℕ := 800  -- Amount received by second truck

/-- The theorem stating the cost of each sack of flour --/
theorem flour_sack_cost (scenario : FlourScenario) : scenario.sack_cost = 1600 :=
  by
    have h1 : scenario.sack_cost * scenario.truck1_left + scenario.truck1_pay = 
              scenario.customs_duty * (scenario.truck1_sacks - scenario.truck1_left) := by sorry
    have h2 : scenario.sack_cost * scenario.truck2_left - scenario.truck2_receive = 
              scenario.customs_duty * (scenario.truck2_sacks - scenario.truck2_left) := by sorry
    sorry  -- The proof goes here

end NUMINAMATH_CALUDE_flour_sack_cost_l2678_267877


namespace NUMINAMATH_CALUDE_subset_sum_divisible_by_p_l2678_267899

theorem subset_sum_divisible_by_p (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  let S := Finset.range (2 * p)
  (S.powerset.filter (fun A => A.card = p ∧ (A.sum id) % p = 0)).card =
    (Nat.choose (2 * p) p - 2) / p + 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_sum_divisible_by_p_l2678_267899
