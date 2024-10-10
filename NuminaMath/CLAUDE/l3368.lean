import Mathlib

namespace shaded_area_of_carpet_l3368_336890

/-- Given a square carpet with side length 12 feet, one large shaded square,
    and twelve smaller congruent shaded squares, where the ratios of side lengths
    are as specified, the total shaded area is 15.75 square feet. -/
theorem shaded_area_of_carpet (S T : ℝ) : 
  (12 : ℝ) / S = 4 →
  S / T = 4 →
  S^2 + 12 * T^2 = 15.75 := by
  sorry

end shaded_area_of_carpet_l3368_336890


namespace certain_amount_proof_l3368_336820

theorem certain_amount_proof : 
  let x : ℝ := 900
  let A : ℝ := 0.15 * 1600 - 0.25 * x
  A = 15 := by sorry

end certain_amount_proof_l3368_336820


namespace new_cost_relation_l3368_336832

/-- Represents the manufacturing cost function -/
def cost (k t b : ℝ) : ℝ := k * (t * b) ^ 4

/-- Theorem: New cost after doubling batches and reducing time by 25% -/
theorem new_cost_relation (k t b : ℝ) (h_pos : t > 0 ∧ b > 0) :
  cost k (0.75 * t) (2 * b) = 25.62890625 * cost k t b := by
  sorry

#check new_cost_relation

end new_cost_relation_l3368_336832


namespace tangent_line_properties_l3368_336864

def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 1 = 0

def line_l (α : ℝ) (x y : ℝ) : Prop := 
  ∃ t : ℝ, x = 4 + t * Real.sin α ∧ y = t * Real.cos α

def is_tangent (α : ℝ) : Prop :=
  ∃ x y : ℝ, curve_C x y ∧ line_l α x y ∧
  ∀ x' y' : ℝ, curve_C x' y' ∧ line_l α x' y' → (x', y') = (x, y)

theorem tangent_line_properties :
  ∀ α : ℝ, 0 ≤ α ∧ α < Real.pi → is_tangent α →
    α = Real.pi / 6 ∧
    ∃ x y : ℝ, curve_C x y ∧ line_l α x y ∧ x = 7/2 ∧ y = -Real.sqrt 3 / 2 :=
by sorry

end tangent_line_properties_l3368_336864


namespace exists_fib_divisible_by_2014_l3368_336802

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: There exists a positive integer n such that F_n is divisible by 2014 -/
theorem exists_fib_divisible_by_2014 : ∃ n : ℕ, n > 0 ∧ 2014 ∣ fib n := by
  sorry

end exists_fib_divisible_by_2014_l3368_336802


namespace wrapping_paper_area_formula_l3368_336861

/-- The area of wrapping paper required for a box -/
def wrapping_paper_area (w : ℝ) (h : ℝ) : ℝ :=
  (4 * w + h) * (2 * w + h)

/-- Theorem: The area of the wrapping paper for a box with width w, length 2w, and height h -/
theorem wrapping_paper_area_formula (w : ℝ) (h : ℝ) :
  wrapping_paper_area w h = 8 * w^2 + 6 * w * h + h^2 := by
  sorry

end wrapping_paper_area_formula_l3368_336861


namespace james_tv_watching_time_l3368_336817

/-- The duration of a Jeopardy episode in minutes -/
def jeopardy_duration : ℕ := 20

/-- The number of Jeopardy episodes watched -/
def jeopardy_episodes : ℕ := 2

/-- The duration of a Wheel of Fortune episode in minutes -/
def wheel_of_fortune_duration : ℕ := 2 * jeopardy_duration

/-- The number of Wheel of Fortune episodes watched -/
def wheel_of_fortune_episodes : ℕ := 2

/-- The total time spent watching TV in minutes -/
def total_time_minutes : ℕ := 
  jeopardy_duration * jeopardy_episodes + 
  wheel_of_fortune_duration * wheel_of_fortune_episodes

/-- Conversion factor from minutes to hours -/
def minutes_per_hour : ℕ := 60

theorem james_tv_watching_time : 
  total_time_minutes / minutes_per_hour = 2 := by sorry

end james_tv_watching_time_l3368_336817


namespace largest_divisor_for_multiples_of_three_l3368_336837

def f (n : ℕ) : ℕ := n * (n + 2) * (n + 4) * (n + 6) * (n + 8)

theorem largest_divisor_for_multiples_of_three :
  ∃ (d : ℕ), d = 288 ∧
  (∀ (n : ℕ), 3 ∣ n → d ∣ f n) ∧
  (∀ (m : ℕ), m > d → ∃ (n : ℕ), 3 ∣ n ∧ ¬(m ∣ f n)) :=
sorry

end largest_divisor_for_multiples_of_three_l3368_336837


namespace tower_height_calculation_l3368_336853

-- Define the tower and measurement points
structure Tower :=
  (height : ℝ)

structure MeasurementPoints :=
  (distanceAD : ℝ)
  (angleA : ℝ)
  (angleD : ℝ)

-- Define the theorem
theorem tower_height_calculation (t : Tower) (m : MeasurementPoints) 
  (h_distanceAD : m.distanceAD = 129)
  (h_angleA : m.angleA = 45)
  (h_angleD : m.angleD = 60) :
  t.height = 305 := by
  sorry


end tower_height_calculation_l3368_336853


namespace expression_equality_l3368_336860

theorem expression_equality : (5^1003 + 6^1004)^2 - (5^1003 - 6^1004)^2 = 24 * 30^1003 := by
  sorry

end expression_equality_l3368_336860


namespace old_edition_pages_l3368_336880

theorem old_edition_pages (new_edition : ℕ) (h1 : new_edition = 450) 
  (h2 : new_edition = 2 * old_edition - 230) : old_edition = 340 :=
by
  sorry

end old_edition_pages_l3368_336880


namespace assignment_methods_eq_eight_l3368_336883

/-- Represents the number of schools --/
def num_schools : ℕ := 2

/-- Represents the number of student teachers --/
def num_teachers : ℕ := 4

/-- Calculates the number of assignment methods --/
def assignment_methods : ℕ := 
  let a_assignments := num_schools -- A can be assigned to either school
  let b_assignments := num_schools - 1 -- B must be assigned to the other school
  let remaining_assignments := num_schools ^ (num_teachers - 2) -- Remaining 2 teachers can be assigned freely
  a_assignments * b_assignments * remaining_assignments

/-- Theorem stating that the number of assignment methods is 8 --/
theorem assignment_methods_eq_eight : assignment_methods = 8 := by
  sorry

end assignment_methods_eq_eight_l3368_336883


namespace AB_value_l3368_336852

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom on_line : ∃ (y : ℝ), A.2 = y ∧ B.2 = y ∧ C.2 = y ∧ D.2 = y
axiom order : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1
axiom AB_eq_CD : dist A B = dist C D
axiom BC_eq_16 : dist B C = 16
axiom E_not_on_line : E.2 ≠ A.2
axiom BE_eq_CE : dist B E = dist C E
axiom BE_eq_13 : dist B E = 13

-- Define perimeter function
def perimeter (X Y Z : ℝ × ℝ) : ℝ := dist X Y + dist Y Z + dist Z X

-- State the theorem
theorem AB_value : 
  perimeter A E D = 3 * perimeter B E C → 
  dist A B = 34/3 :=
sorry

end AB_value_l3368_336852


namespace right_triangle_with_53_hypotenuse_l3368_336826

theorem right_triangle_with_53_hypotenuse (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 53 →           -- Hypotenuse is 53
  b = a + 1 →        -- Legs are consecutive integers
  a + b = 75 :=      -- Sum of legs is 75
by sorry

end right_triangle_with_53_hypotenuse_l3368_336826


namespace square_polynomial_k_values_l3368_336873

theorem square_polynomial_k_values (k : ℝ) : 
  (∃ p : ℝ → ℝ, ∀ x, x^2 + 2*(k-1)*x + 64 = (p x)^2) → 
  (k = 9 ∨ k = -7) := by
  sorry

end square_polynomial_k_values_l3368_336873


namespace simplify_expression_l3368_336868

theorem simplify_expression (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 2*b) (h3 : a ≠ b) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) :=
by sorry

end simplify_expression_l3368_336868


namespace bryans_mineral_samples_per_shelf_l3368_336834

/-- Given Bryan's mineral collection setup, prove the number of samples per shelf. -/
theorem bryans_mineral_samples_per_shelf :
  let total_samples : ℕ := 455
  let total_shelves : ℕ := 7
  let samples_per_shelf : ℕ := total_samples / total_shelves
  samples_per_shelf = 65 := by
  sorry

end bryans_mineral_samples_per_shelf_l3368_336834


namespace sqrt_sum_fractions_l3368_336859

theorem sqrt_sum_fractions : 
  Real.sqrt (4/25 + 9/49) = Real.sqrt 421 / 35 := by sorry

end sqrt_sum_fractions_l3368_336859


namespace total_tomatoes_l3368_336889

/-- The number of cucumber rows for each tomato row -/
def cucumber_rows_per_tomato_row : ℕ := 2

/-- The total number of rows in the garden -/
def total_rows : ℕ := 15

/-- The number of tomato plants in each row -/
def plants_per_row : ℕ := 8

/-- The number of tomatoes produced by each plant -/
def tomatoes_per_plant : ℕ := 3

/-- The theorem stating the total number of tomatoes Aubrey will have -/
theorem total_tomatoes : 
  (total_rows / (cucumber_rows_per_tomato_row + 1)) * plants_per_row * tomatoes_per_plant = 120 := by
  sorry

end total_tomatoes_l3368_336889


namespace adams_apples_l3368_336848

/-- 
Given:
- Jackie has 10 apples
- Jackie has 2 more apples than Adam
Prove that Adam has 8 apples
-/
theorem adams_apples (jackie_apples : ℕ) (adam_apples : ℕ) 
  (h1 : jackie_apples = 10)
  (h2 : jackie_apples = adam_apples + 2) : 
  adam_apples = 8 := by
  sorry

end adams_apples_l3368_336848


namespace sin_225_degrees_l3368_336843

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_225_degrees_l3368_336843


namespace cube_lines_properties_l3368_336813

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  a : ℝ
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Defines a cube with given edge length and correct vertex positions -/
def makeCube (a : ℝ) : Cube := {
  a := a,
  A := ⟨0, 0, 0⟩,
  B := ⟨a, 0, 0⟩,
  C := ⟨a, a, 0⟩,
  D := ⟨0, a, 0⟩,
  A₁ := ⟨0, 0, a⟩,
  B₁ := ⟨a, 0, a⟩,
  C₁ := ⟨a, a, a⟩,
  D₁ := ⟨0, a, a⟩
}

/-- Calculates the angle between two lines in the cube -/
def angleBetweenLines (cube : Cube) : ℝ := sorry

/-- Calculates the distance between two lines in the cube -/
def distanceBetweenLines (cube : Cube) : ℝ := sorry

theorem cube_lines_properties (a : ℝ) (h : a > 0) :
  let cube := makeCube a
  angleBetweenLines cube = 90 ∧ 
  distanceBetweenLines cube = a * Real.sqrt 6 / 6 := by
  sorry

end cube_lines_properties_l3368_336813


namespace gcd_150_450_l3368_336836

theorem gcd_150_450 : Nat.gcd 150 450 = 150 := by
  sorry

end gcd_150_450_l3368_336836


namespace no_three_primes_sum_squares_l3368_336849

theorem no_three_primes_sum_squares : ¬∃ (p q r : ℕ), 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  ∃ (a b c : ℕ), p + q = a^2 ∧ p + r = b^2 ∧ q + r = c^2 :=
by sorry

end no_three_primes_sum_squares_l3368_336849


namespace no_prime_sum_10003_l3368_336829

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 10003 := by
  sorry

end no_prime_sum_10003_l3368_336829


namespace expression_simplification_l3368_336892

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 2) :
  (1 + 1 / (m - 2)) / ((m^2 - m) / (m - 2)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l3368_336892


namespace min_value_of_sum_min_value_is_nine_min_value_achieved_l3368_336833

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → 4*a + b ≤ 4*x + y :=
by sorry

theorem min_value_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  4*a + b ≥ 9 :=
by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 ∧ 4*x + y = 9 :=
by sorry

end min_value_of_sum_min_value_is_nine_min_value_achieved_l3368_336833


namespace parity_of_A_15_16_17_l3368_336893

def A : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => A (n + 2) + A n

theorem parity_of_A_15_16_17 : 
  Odd (A 15) ∧ Even (A 16) ∧ Odd (A 17) := by sorry

end parity_of_A_15_16_17_l3368_336893


namespace john_average_bottle_price_l3368_336876

/-- The average price of bottles purchased by John -/
def average_price (large_quantity : ℕ) (large_price : ℚ) (small_quantity : ℕ) (small_price : ℚ) : ℚ :=
  (large_quantity * large_price + small_quantity * small_price) / (large_quantity + small_quantity)

/-- The average price of bottles purchased by John is approximately $1.70 -/
theorem john_average_bottle_price :
  let large_quantity : ℕ := 1300
  let large_price : ℚ := 189/100
  let small_quantity : ℕ := 750
  let small_price : ℚ := 138/100
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
    |average_price large_quantity large_price small_quantity small_price - 17/10| < ε :=
sorry

end john_average_bottle_price_l3368_336876


namespace systematic_sample_result_l3368_336878

def systematic_sample (total : ℕ) (sample_size : ℕ) (interval_start : ℕ) (interval_end : ℕ) : ℕ :=
  let sample_interval := total / sample_size
  let interval_size := interval_end - interval_start + 1
  interval_size / sample_interval

theorem systematic_sample_result :
  systematic_sample 360 20 181 288 = 6 := by
  sorry

end systematic_sample_result_l3368_336878


namespace certain_number_equation_l3368_336867

theorem certain_number_equation : ∃ x : ℚ, (40 * x + (12 + 8) * 3 / 5 = 1212) ∧ x = 30 := by
  sorry

end certain_number_equation_l3368_336867


namespace no_rectangular_parallelepiped_sum_866_l3368_336854

theorem no_rectangular_parallelepiped_sum_866 :
  ¬∃ (x y z : ℕ+), x * y * z + 2 * (x * y + x * z + y * z) + 4 * (x + y + z) = 866 := by
sorry

end no_rectangular_parallelepiped_sum_866_l3368_336854


namespace zero_points_sum_inequality_l3368_336877

theorem zero_points_sum_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂)
  (h₄ : Real.log x₁ - a * x₁ = 0) (h₅ : Real.log x₂ - a * x₂ = 0) :
  x₁ + x₂ > 2 / a :=
by sorry

end zero_points_sum_inequality_l3368_336877


namespace max_initial_states_l3368_336847

/-- Represents the state of friendships between sheep and the wolf -/
structure SheepcoteState (n : ℕ) :=
  (wolf_friends : Finset (Fin n))
  (sheep_friendships : Finset (Fin n × Fin n))

/-- Represents the process of the wolf eating sheep -/
def eat_sheep (state : SheepcoteState n) : Option (SheepcoteState n) :=
  sorry

/-- Checks if all sheep can be eaten given an initial state -/
def can_eat_all_sheep (initial_state : SheepcoteState n) : Prop :=
  sorry

/-- The number of valid initial states -/
def num_valid_initial_states (n : ℕ) : ℕ :=
  sorry

theorem max_initial_states (n : ℕ) :
  num_valid_initial_states n = 2^(n-1) :=
sorry

end max_initial_states_l3368_336847


namespace sequence_equality_l3368_336856

-- Define the sequence a_n
def a (n : ℕ) (x : ℝ) : ℝ := 1 + x^(n+1) + x^(n+2)

-- State the theorem
theorem sequence_equality (x : ℝ) (h : (a 2 x)^2 = (a 1 x) * (a 3 x)) :
  ∀ n ≥ 3, (a n x)^2 = (a (n-1) x) * (a (n+1) x) :=
by sorry

end sequence_equality_l3368_336856


namespace max_value_of_sum_products_l3368_336831

theorem max_value_of_sum_products (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_constraint : a + b + c + d = 200) :
  ab + bc + cd ≤ 10000 :=
sorry

end max_value_of_sum_products_l3368_336831


namespace diagonals_150_sided_polygon_l3368_336845

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_150_sided_polygon : num_diagonals 150 = 11025 := by
  sorry

end diagonals_150_sided_polygon_l3368_336845


namespace vanessa_points_l3368_336814

/-- Calculates the points scored by a player given the total team points,
    number of other players, and average points of other players. -/
def player_points (total_points : ℕ) (other_players : ℕ) (avg_other_points : ℚ) : ℚ :=
  total_points - other_players * avg_other_points

/-- Proves that Vanessa scored 27 points given the problem conditions. -/
theorem vanessa_points :
  let total_points : ℕ := 48
  let other_players : ℕ := 6
  let avg_other_points : ℚ := 7/2
  player_points total_points other_players avg_other_points = 27 := by
sorry

#eval player_points 48 6 (7/2)

end vanessa_points_l3368_336814


namespace complex_equation_implies_sum_of_squares_l3368_336899

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Given complex equation -/
def complex_equation (a b : ℝ) : Prop :=
  (4 - 3 * i) * (a + b * i) = 25 * i

/-- Theorem stating that the given complex equation implies a² + b² = 25 -/
theorem complex_equation_implies_sum_of_squares (a b : ℝ) :
  complex_equation a b → a^2 + b^2 = 25 := by
  sorry

end complex_equation_implies_sum_of_squares_l3368_336899


namespace remainder_problem_l3368_336897

theorem remainder_problem (x : ℤ) : x % 82 = 5 → (x + 17) % 41 = 22 := by
  sorry

end remainder_problem_l3368_336897


namespace unique_integer_sqrt_l3368_336862

theorem unique_integer_sqrt (x y : ℕ) : x = 25530 ∧ y = 29464 ↔ 
  ∃ (z : ℕ), z > 0 ∧ z * z = x * x + y * y ∧
  ∀ (a b : ℕ), (a = 37615 ∧ b = 26855) ∨ 
               (a = 15123 ∧ b = 32477) ∨ 
               (a = 28326 ∧ b = 28614) ∨ 
               (a = 22536 ∧ b = 27462) →
               ¬∃ (w : ℕ), w > 0 ∧ w * w = a * a + b * b :=
by sorry

end unique_integer_sqrt_l3368_336862


namespace quadratic_positive_combination_l3368_336857

/-- A quadratic function is a function of the form ax² + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- Two intervals are disjoint if they have no common points -/
def DisjointIntervals (I J : Set ℝ) : Prop :=
  I ∩ J = ∅

/-- A function is negative on an interval if it takes negative values for all points in that interval -/
def NegativeOnInterval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, f x < 0

theorem quadratic_positive_combination
  (f g : ℝ → ℝ)
  (hf : IsQuadratic f)
  (hg : IsQuadratic g)
  (hfI : ∃ I : Set ℝ, NegativeOnInterval f I)
  (hgJ : ∃ J : Set ℝ, NegativeOnInterval g J)
  (hIJ : ∀ I J, (NegativeOnInterval f I ∧ NegativeOnInterval g J) → DisjointIntervals I J) :
  ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ ∀ x, α * f x + β * g x > 0 :=
sorry

end quadratic_positive_combination_l3368_336857


namespace eli_age_difference_l3368_336882

/-- Given the ages and relationships between Kaylin, Sarah, Eli, and Freyja, prove that Eli is 9 years older than Freyja. -/
theorem eli_age_difference (kaylin sarah eli freyja : ℕ) : 
  kaylin = 33 →
  freyja = 10 →
  sarah = kaylin + 5 →
  sarah = 2 * eli →
  eli > freyja →
  eli - freyja = 9 := by
  sorry

end eli_age_difference_l3368_336882


namespace fraction_sum_integer_implies_fractions_integer_l3368_336818

theorem fraction_sum_integer_implies_fractions_integer (x y : ℕ) :
  ∃ k : ℤ, (x^2 - 1 : ℚ) / (y + 1) + (y^2 - 1 : ℚ) / (x + 1) = k →
  ∃ m n : ℤ, (x^2 - 1 : ℚ) / (y + 1) = m ∧ (y^2 - 1 : ℚ) / (x + 1) = n :=
by sorry

end fraction_sum_integer_implies_fractions_integer_l3368_336818


namespace parabola_line_theorem_l3368_336888

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the perpendicular bisector of a line with slope m
def perpendicular_bisector (m : ℝ) (x y : ℝ) : Prop := x = -1/m * y + (2*m^2 + 3)

-- Define the condition for points to lie on the same circle
def on_same_circle (A B M N : ℝ × ℝ) : Prop := 
  let midpoint_AB := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  let midpoint_MN := ((M.1 + N.1)/2, (M.2 + N.2)/2)
  (A.1 - midpoint_MN.1)^2 + (A.2 - midpoint_MN.2)^2 = 
  (M.1 - midpoint_AB.1)^2 + (M.2 - midpoint_AB.2)^2

theorem parabola_line_theorem (m : ℝ) :
  ∃ A B M N : ℝ × ℝ,
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ parabola M.1 M.2 ∧ parabola N.1 N.2 ∧
    line_through_focus m A.1 A.2 ∧ line_through_focus m B.1 B.2 ∧
    perpendicular_bisector m M.1 M.2 ∧ perpendicular_bisector m N.1 N.2 ∧
    on_same_circle A B M N →
  m = 1 ∨ m = -1 := by sorry

end parabola_line_theorem_l3368_336888


namespace polynomial_remainder_l3368_336815

theorem polynomial_remainder (x : ℝ) : 
  let p := fun x => 5*x^4 - 9*x^3 + 3*x^2 - 7*x - 30
  let d := fun x => 3*x - 9
  p 3 = 138 := by
  sorry

end polynomial_remainder_l3368_336815


namespace max_sum_of_products_l3368_336828

/-- Represents the assignment of numbers to cube faces -/
def CubeAssignment := Fin 6 → Fin 6

/-- Computes the sum of products at cube vertices given a face assignment -/
def sumOfProducts (assignment : CubeAssignment) : ℕ :=
  sorry

/-- The set of all possible cube assignments -/
def allAssignments : Set CubeAssignment :=
  sorry

theorem max_sum_of_products :
  ∃ (assignment : CubeAssignment),
    assignment ∈ allAssignments ∧
    sumOfProducts assignment = 343 ∧
    ∀ (other : CubeAssignment),
      other ∈ allAssignments →
      sumOfProducts other ≤ 343 :=
sorry

end max_sum_of_products_l3368_336828


namespace expression_equality_l3368_336840

theorem expression_equality : (2^5 * 9^2) / (8^2 * 3^5) = 1/6 := by
  sorry

end expression_equality_l3368_336840


namespace line_vector_proof_l3368_336811

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 1 = (2, -3, 5) ∧ line_vector 4 = (-2, 9, -11)) →
  line_vector 5 = (-10/3, 13, -49/3) := by
  sorry

end line_vector_proof_l3368_336811


namespace uniqueness_not_algorithm_characteristic_l3368_336821

/-- Represents characteristics of an algorithm -/
inductive AlgorithmCharacteristic
  | Abstraction
  | Precision
  | Finiteness
  | Uniqueness

/-- Predicate to check if a given characteristic is a valid algorithm characteristic -/
def isValidAlgorithmCharacteristic (c : AlgorithmCharacteristic) : Prop :=
  match c with
  | AlgorithmCharacteristic.Abstraction => True
  | AlgorithmCharacteristic.Precision => True
  | AlgorithmCharacteristic.Finiteness => True
  | AlgorithmCharacteristic.Uniqueness => False

theorem uniqueness_not_algorithm_characteristic :
  ¬(isValidAlgorithmCharacteristic AlgorithmCharacteristic.Uniqueness) :=
by sorry

end uniqueness_not_algorithm_characteristic_l3368_336821


namespace solution_count_l3368_336808

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem solution_count :
  let gcd_value := factorial 20
  let lcm_value := factorial 30
  (Finset.filter (fun p : ℕ × ℕ => 
    Nat.gcd p.1 p.2 = gcd_value ∧ 
    Nat.lcm p.1 p.2 = lcm_value
  ) (Finset.product (Finset.range (lcm_value + 1)) (Finset.range (lcm_value + 1)))).card = 1024 := by
  sorry

end solution_count_l3368_336808


namespace bakers_sales_l3368_336844

/-- Baker's cake and pastry problem -/
theorem bakers_sales (cakes_made pastries_made pastries_sold : ℕ) 
  (h1 : cakes_made = 157)
  (h2 : pastries_made = 169)
  (h3 : pastries_sold = 147)
  (h4 : ∃ cakes_sold : ℕ, cakes_sold = pastries_sold + 11) :
  ∃ cakes_sold : ℕ, cakes_sold = 158 := by
  sorry

end bakers_sales_l3368_336844


namespace odd_square_minus_one_div_eight_l3368_336885

theorem odd_square_minus_one_div_eight (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^2 - 1 = 8*k := by sorry

end odd_square_minus_one_div_eight_l3368_336885


namespace girls_from_clay_l3368_336839

/-- Represents a school in the science camp --/
inductive School
| Jonas
| Clay
| Maple

/-- Represents the gender of a student --/
inductive Gender
| Boy
| Girl

/-- Represents the distribution of students in the science camp --/
structure CampDistribution where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  jonas_students : ℕ
  clay_students : ℕ
  maple_students : ℕ
  jonas_boys : ℕ

/-- The actual distribution of students in the science camp --/
def camp_distribution : CampDistribution :=
  { total_students := 120
  , total_boys := 70
  , total_girls := 50
  , jonas_students := 50
  , clay_students := 40
  , maple_students := 30
  , jonas_boys := 30
  }

/-- Theorem stating that the number of girls from Clay Middle School is 10 --/
theorem girls_from_clay (d : CampDistribution) (h : d = camp_distribution) :
  ∃ (clay_girls : ℕ), clay_girls = 10 ∧
  clay_girls = d.clay_students - (d.total_boys - d.jonas_boys) :=
by sorry

end girls_from_clay_l3368_336839


namespace sqrt_sum_equals_two_l3368_336812

theorem sqrt_sum_equals_two : 
  Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 2 := by
  sorry

end sqrt_sum_equals_two_l3368_336812


namespace rectangle_square_ratio_l3368_336823

/-- Represents a right triangle with a rectangle and square inscribed as described in the problem -/
structure TriangleWithInscriptions where
  /-- Side lengths of the right triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Side lengths of the inscribed rectangle -/
  rect_side1 : ℝ
  rect_side2 : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- Conditions for the triangle -/
  triangle_right : a^2 + b^2 = c^2
  triangle_sides : a = 5 ∧ b = 12 ∧ c = 13
  /-- Conditions for the rectangle -/
  rectangle_sides : rect_side1 = 5 ∧ rect_side2 = 12
  /-- Condition for the square -/
  square_formula : square_side = (a * b) / c

/-- The main theorem stating the ratio of the longer rectangle side to the square side -/
theorem rectangle_square_ratio (t : TriangleWithInscriptions) :
  t.rect_side2 / t.square_side = 13 / 5 := by
  sorry

end rectangle_square_ratio_l3368_336823


namespace zero_point_in_interval_l3368_336874

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^(x-2)

theorem zero_point_in_interval :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ 1 < x₀ ∧ x₀ < 2 :=
sorry

end zero_point_in_interval_l3368_336874


namespace kate_bouncy_balls_l3368_336846

/-- The number of packs of yellow bouncy balls Kate bought -/
def yellow_packs : ℕ := 6

/-- The number of packs of red bouncy balls Kate bought -/
def red_packs : ℕ := 7

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 18

theorem kate_bouncy_balls :
  yellow_packs * balls_per_pack + 18 = red_packs * balls_per_pack :=
by sorry

end kate_bouncy_balls_l3368_336846


namespace hyperbola_asymptote_slope_l3368_336898

/-- The slope of the asymptotes for the hyperbola (x^2 / 144) - (y^2 / 81) = 1 is 3/4 -/
theorem hyperbola_asymptote_slope :
  let hyperbola := fun (x y : ℝ) => x^2 / 144 - y^2 / 81 = 1
  ∃ m : ℝ, m = 3/4 ∧ 
    ∀ (x y : ℝ), hyperbola x y → (y = m * x ∨ y = -m * x) := by
  sorry

end hyperbola_asymptote_slope_l3368_336898


namespace probability_of_y_selection_l3368_336822

theorem probability_of_y_selection (p_x p_both : ℝ) (h1 : p_x = 1/3) (h2 : p_both = 0.13333333333333333) : 
  ∃ p_y : ℝ, p_y = 0.4 ∧ p_both = p_x * p_y :=
sorry

end probability_of_y_selection_l3368_336822


namespace fruit_basket_count_l3368_336830

/-- The number of ways to choose items from a set of n items -/
def choiceCount (n : ℕ) : ℕ := n + 1

/-- The total number of fruit baskets including the empty basket -/
def totalBaskets (appleCount orangeCount : ℕ) : ℕ :=
  choiceCount appleCount * choiceCount orangeCount

/-- The number of valid fruit baskets (excluding the empty basket) -/
def validBaskets (appleCount orangeCount : ℕ) : ℕ :=
  totalBaskets appleCount orangeCount - 1

theorem fruit_basket_count :
  validBaskets 7 12 = 103 := by
  sorry

end fruit_basket_count_l3368_336830


namespace x_value_l3368_336879

theorem x_value : ∃ x : ℚ, (3 * x - 2) / 7 = 15 ∧ x = 107 / 3 := by
  sorry

end x_value_l3368_336879


namespace supermarket_sales_problem_l3368_336891

/-- Represents the monthly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -5 * x + 550

/-- Represents the monthly profit as a function of selling price -/
def monthly_profit (x : ℝ) : ℝ := sales_volume x * (x - 50)

/-- The cost per item -/
def cost : ℝ := 50

/-- The initial selling price -/
def initial_price : ℝ := 100

/-- The initial monthly sales -/
def initial_sales : ℝ := 50

/-- The change in sales for every 2 yuan decrease in price -/
def sales_change : ℝ := 10

theorem supermarket_sales_problem :
  (∀ x : ℝ, x ≥ cost → sales_volume x = -5 * x + 550) ∧
  (∃ x : ℝ, x ≥ cost ∧ monthly_profit x = 4000 ∧ x = 70) ∧
  (∃ x : ℝ, x ≥ cost ∧ ∀ y : ℝ, y ≥ cost → monthly_profit x ≥ monthly_profit y ∧ x = 80) :=
by sorry

end supermarket_sales_problem_l3368_336891


namespace fibonacci_like_sequence_l3368_336886

theorem fibonacci_like_sequence
  (a : ℕ → ℕ)  -- Sequence of natural numbers
  (seq_length : ℕ)  -- Length of the sequence
  (h_length : seq_length = 10)  -- The sequence has 10 numbers
  (h_rec : ∀ n, 3 ≤ n → n ≤ seq_length → a n = a (n-1) + a (n-2))  -- Recurrence relation
  (h_seventh : a 7 = 42)  -- The seventh number is 42
  (h_ninth : a 9 = 110)  -- The ninth number is 110
  : a 4 = 10 :=  -- The fourth number is 10
by sorry

end fibonacci_like_sequence_l3368_336886


namespace arithmetic_geometric_mean_problem_l3368_336825

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 96) :
  x^2 + y^2 = 1408 := by
  sorry

end arithmetic_geometric_mean_problem_l3368_336825


namespace smallest_number_divisible_by_primes_l3368_336896

def first_17_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

def is_divisible_by_all_except_two_consecutive (n : Nat) (primes : List Nat) (i : Nat) : Prop :=
  ∀ (p : Nat), p ∈ primes → (p ≠ primes[i]! ∧ p ≠ primes[i+1]!) → n % p = 0

theorem smallest_number_divisible_by_primes : ∃ (n : Nat),
  is_divisible_by_all_except_two_consecutive n first_17_primes 15 ∧
  ∀ (m : Nat), m < n → ¬is_divisible_by_all_except_two_consecutive m first_17_primes 15 :=
sorry

end smallest_number_divisible_by_primes_l3368_336896


namespace apples_difference_l3368_336881

/-- The number of apples Yanna bought -/
def total_apples : ℕ := 60

/-- The number of apples Yanna gave to Zenny -/
def apples_to_zenny : ℕ := 18

/-- The number of apples Yanna kept for herself -/
def apples_kept : ℕ := 36

/-- The number of apples Yanna gave to Andrea -/
def apples_to_andrea : ℕ := total_apples - apples_to_zenny - apples_kept

theorem apples_difference :
  apples_to_zenny - apples_to_andrea = 12 :=
by sorry

end apples_difference_l3368_336881


namespace expression_simplification_l3368_336827

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 36 := by
  sorry

end expression_simplification_l3368_336827


namespace square_perimeter_47_20_l3368_336875

-- Define the side length of the square
def side_length : ℚ := 47 / 20

-- Define the perimeter of a square
def square_perimeter (s : ℚ) : ℚ := 4 * s

-- Theorem statement
theorem square_perimeter_47_20 : 
  square_perimeter side_length = 47 / 5 := by
  sorry

end square_perimeter_47_20_l3368_336875


namespace last_digit_power_sum_l3368_336842

theorem last_digit_power_sum (m : ℕ+) : (2^(m.val + 2007) + 2^(m.val + 1)) % 10 = 0 := by
  sorry

end last_digit_power_sum_l3368_336842


namespace exists_real_less_than_negative_one_l3368_336855

theorem exists_real_less_than_negative_one : ∃ x : ℝ, x < -1 := by
  sorry

end exists_real_less_than_negative_one_l3368_336855


namespace complex_sum_direction_l3368_336884

theorem complex_sum_direction (r : ℝ) (h : r > 0) :
  ∃ (r : ℝ), r > 0 ∧ 
  Complex.exp (11 * Real.pi * Complex.I / 60) +
  Complex.exp (21 * Real.pi * Complex.I / 60) +
  Complex.exp (31 * Real.pi * Complex.I / 60) +
  Complex.exp (41 * Real.pi * Complex.I / 60) +
  Complex.exp (51 * Real.pi * Complex.I / 60) =
  r * Complex.exp (31 * Real.pi * Complex.I / 60) :=
by sorry

end complex_sum_direction_l3368_336884


namespace book_distribution_ways_l3368_336870

/-- The number of ways to distribute identical books between two states --/
def distribute_books (n : ℕ) : ℕ := n - 1

/-- The number of books --/
def total_books : ℕ := 8

/-- Theorem: The number of ways to distribute eight identical books between
    the library and being checked out, with at least one book in each state,
    is equal to 7. --/
theorem book_distribution_ways :
  distribute_books total_books = 7 := by
  sorry

end book_distribution_ways_l3368_336870


namespace sin_squared_minus_two_sin_range_l3368_336866

theorem sin_squared_minus_two_sin_range :
  ∀ x : ℝ, -1 ≤ Real.sin x ^ 2 - 2 * Real.sin x ∧ Real.sin x ^ 2 - 2 * Real.sin x ≤ 3 := by
  sorry

end sin_squared_minus_two_sin_range_l3368_336866


namespace derivative_at_one_l3368_336895

-- Define the function
def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 4 := by sorry

end derivative_at_one_l3368_336895


namespace arithmetic_sequence_problem_l3368_336871

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, a (n + 1) - a n = 1) →  -- Common difference is 1
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Sum formula for arithmetic sequence
  (S 6 = 4 * S 3) →  -- Given condition
  a 10 = 19 / 2 := by
sorry

end arithmetic_sequence_problem_l3368_336871


namespace quadratic_form_minimum_l3368_336841

theorem quadratic_form_minimum (x y : ℝ) :
  3 * x^2 + 4 * x * y + y^2 - 6 * x + 2 * y + 9 ≥ 9/4 ∧
  (3 * x^2 + 4 * x * y + y^2 - 6 * x + 2 * y + 9 = 9/4 ↔ x = 3/2 ∧ y = -3/4) := by
sorry

end quadratic_form_minimum_l3368_336841


namespace concert_duration_13h25m_l3368_336810

/-- Calculates the total duration in minutes of a concert given its length in hours and minutes. -/
def concert_duration (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

/-- Theorem stating that a concert lasting 13 hours and 25 minutes has a total duration of 805 minutes. -/
theorem concert_duration_13h25m :
  concert_duration 13 25 = 805 := by
  sorry

end concert_duration_13h25m_l3368_336810


namespace sin_30_degrees_l3368_336819

/-- Sine of 30 degrees is equal to 1/2 -/
theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_30_degrees_l3368_336819


namespace trip_time_difference_l3368_336816

-- Define the given conditions
def speed : ℝ := 60
def distance1 : ℝ := 510
def distance2 : ℝ := 540

-- Define the theorem
theorem trip_time_difference : 
  (distance2 - distance1) / speed * 60 = 30 :=
by
  sorry

end trip_time_difference_l3368_336816


namespace negative_three_point_fourteen_greater_than_negative_pi_l3368_336838

theorem negative_three_point_fourteen_greater_than_negative_pi : -3.14 > -π := by
  sorry

end negative_three_point_fourteen_greater_than_negative_pi_l3368_336838


namespace anne_found_five_bottle_caps_l3368_336800

/-- The number of bottle caps Anne found -/
def bottle_caps_found (initial final : ℕ) : ℕ := final - initial

/-- Proof that Anne found 5 bottle caps -/
theorem anne_found_five_bottle_caps :
  bottle_caps_found 10 15 = 5 := by
  sorry

end anne_found_five_bottle_caps_l3368_336800


namespace possible_values_of_a_l3368_336805

def P : Set ℝ := {x | x^2 + x - 6 = 0}
def S (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (S a ⊆ P) ↔ (a = 0 ∨ a = 1/3 ∨ a = -1/2) :=
by sorry

end possible_values_of_a_l3368_336805


namespace certain_expression_l3368_336851

theorem certain_expression (a b X : ℝ) 
  (h1 : a / b = 4 / 3) 
  (h2 : (3 * a + 2 * b) / X = 3) : 
  X = 2 * b := by
sorry

end certain_expression_l3368_336851


namespace unique_integers_square_sum_l3368_336865

theorem unique_integers_square_sum : ∃! (A B : ℕ), 
  A ≤ 9 ∧ B ≤ 9 ∧ (1001 * A + 110 * B)^2 = 57108249 ∧ 10 * A + B = 75 := by
  sorry

end unique_integers_square_sum_l3368_336865


namespace determinant_of_specific_matrix_l3368_336807

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -3, 6]
  Matrix.det A = 36 := by
  sorry

end determinant_of_specific_matrix_l3368_336807


namespace wilson_fraction_problem_l3368_336824

theorem wilson_fraction_problem (N F : ℚ) : 
  N = 8 → N - F * N = 16/3 → F = 1/3 := by
  sorry

end wilson_fraction_problem_l3368_336824


namespace scientific_notation_conversion_l3368_336809

theorem scientific_notation_conversion :
  (1.8 : ℝ) * (10 ^ 8) = 180000000 := by
  sorry

end scientific_notation_conversion_l3368_336809


namespace heart_then_club_probability_l3368_336858

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def numHearts : ℕ := 13

/-- Number of clubs in a standard deck -/
def numClubs : ℕ := 13

/-- Probability of drawing a heart followed by a club from a standard deck -/
def probHeartThenClub : ℚ := numHearts / standardDeck * numClubs / (standardDeck - 1)

theorem heart_then_club_probability :
  probHeartThenClub = 13 / 204 := by sorry

end heart_then_club_probability_l3368_336858


namespace next_two_numbers_l3368_336803

def arithmetic_sequence (n : ℕ) : ℕ := n + 1

theorem next_two_numbers (n : ℕ) (h : n ≥ 6) :
  arithmetic_sequence n = n + 1 ∧
  arithmetic_sequence (n + 1) = n + 2 :=
by sorry

end next_two_numbers_l3368_336803


namespace expression_value_when_b_is_3_l3368_336887

theorem expression_value_when_b_is_3 :
  let b : ℝ := 3
  let expr := (3 * b⁻¹ + b⁻¹ / 3) / b^2
  expr = 10 / 81 := by sorry

end expression_value_when_b_is_3_l3368_336887


namespace bakery_profit_is_175_l3368_336872

/-- Calculates the total profit for Uki's bakery over five days -/
def bakery_profit : ℝ :=
  let cupcake_price : ℝ := 1.50
  let cookie_price : ℝ := 2.00
  let biscuit_price : ℝ := 1.00
  let cupcake_cost : ℝ := 0.75
  let cookie_cost : ℝ := 1.00
  let biscuit_cost : ℝ := 0.50
  let daily_cupcakes : ℝ := 20
  let daily_cookies : ℝ := 10
  let daily_biscuits : ℝ := 20
  let days : ℝ := 5
  let daily_profit : ℝ := 
    (cupcake_price - cupcake_cost) * daily_cupcakes +
    (cookie_price - cookie_cost) * daily_cookies +
    (biscuit_price - biscuit_cost) * daily_biscuits
  daily_profit * days

theorem bakery_profit_is_175 : bakery_profit = 175 := by
  sorry

end bakery_profit_is_175_l3368_336872


namespace linear_function_composition_l3368_336894

theorem linear_function_composition (a b : ℝ) :
  (∀ x y : ℝ, x < y → (a * x + b) < (a * y + b)) →
  (∀ x : ℝ, a * (a * x + b) + b = 4 * x - 1) →
  a = -2 ∧ b = 1 := by
sorry

end linear_function_composition_l3368_336894


namespace find_m_l3368_336801

theorem find_m : ∃ m : ℝ, 
  (∃ y : ℝ, 2 - 3 * (1 - y) = 2 * y) ∧ 
  (∃ x : ℝ, m * (x - 3) - 2 = -8) ∧ 
  (∀ y x : ℝ, 2 - 3 * (1 - y) = 2 * y ↔ m * (x - 3) - 2 = -8) → 
  m = 3 := by
sorry

end find_m_l3368_336801


namespace parallel_lines_k_values_l3368_336804

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line l₁: (k-3)x+(4-k)y+1=0 -/
def l1 (k : ℝ) : Line :=
  { a := k - 3, b := 4 - k, c := 1 }

/-- The second line l₂: 2(k-3)-2y+3=0, rewritten as 2(k-3)x-2y+3=0 -/
def l2 (k : ℝ) : Line :=
  { a := 2 * (k - 3), b := -2, c := 3 }

/-- Theorem stating that if l₁ and l₂ are parallel, then k is either 3 or 5 -/
theorem parallel_lines_k_values :
  ∀ k, parallel (l1 k) (l2 k) → k = 3 ∨ k = 5 := by
  sorry

#check parallel_lines_k_values

end parallel_lines_k_values_l3368_336804


namespace subtraction_puzzle_l3368_336806

theorem subtraction_puzzle (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9)
  (h4 : (100 * a + 10 * b + c) - (100 * c + 10 * b + a) % 10 = 2)
  (h5 : b = c - 1)
  (h6 : (100 * a + 10 * b + c) - (100 * c + 10 * b + a) / 100 = 8) :
  a = 0 ∧ b = 1 ∧ c = 2 := by
  sorry

end subtraction_puzzle_l3368_336806


namespace arithmetic_sequence_problem_l3368_336835

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence)
  (h1 : seq.a 1 + seq.a 3 = 8)
  (h2 : seq.a 4 ^ 2 = seq.a 2 * seq.a 9) :
  seq.a 5 = 13 := by
  sorry

end arithmetic_sequence_problem_l3368_336835


namespace cab_ride_cost_per_mile_l3368_336869

/-- Calculates the cost per mile for Briar's cab rides -/
theorem cab_ride_cost_per_mile
  (days : ℕ)
  (distance_to_event : ℝ)
  (total_cost : ℝ)
  (h1 : days = 7)
  (h2 : distance_to_event = 200)
  (h3 : total_cost = 7000) :
  total_cost / (2 * days * distance_to_event) = 2.5 := by
  sorry

#check cab_ride_cost_per_mile

end cab_ride_cost_per_mile_l3368_336869


namespace perpendicular_lines_m_values_l3368_336850

-- Define the coefficients of the two lines as functions of m
def line1_coeff (m : ℝ) : ℝ × ℝ := (m + 2, m)
def line2_coeff (m : ℝ) : ℝ × ℝ := (m - 1, m - 4)

-- Define the perpendicularity condition
def perpendicular (m : ℝ) : Prop :=
  (line1_coeff m).1 * (line2_coeff m).1 + (line1_coeff m).2 * (line2_coeff m).2 = 0

-- State the theorem
theorem perpendicular_lines_m_values :
  ∀ m : ℝ, perpendicular m → m = -1/2 ∨ m = 2 := by sorry

end perpendicular_lines_m_values_l3368_336850


namespace absolute_value_inequality_l3368_336863

theorem absolute_value_inequality (x : ℝ) : 
  (|x - 2| + |x - 3| < 9) ↔ (-2 < x ∧ x < 7) := by
  sorry

end absolute_value_inequality_l3368_336863
