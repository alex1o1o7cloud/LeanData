import Mathlib

namespace NUMINAMATH_CALUDE_triangle_theorem_l1786_178628

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h : 2 * t.a * Real.sin t.A = (2 * t.b - t.c) * Real.sin t.B + (2 * t.c - t.b) * Real.sin t.C) :
  t.A = Real.pi / 3 ∧ 
  (Real.sin t.B + Real.sin t.C = Real.sqrt 3 → t.A = t.B ∧ t.B = t.C) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1786_178628


namespace NUMINAMATH_CALUDE_marie_glue_sticks_l1786_178612

theorem marie_glue_sticks :
  ∀ (allison_glue allison_paper marie_glue marie_paper : ℕ),
    allison_glue = marie_glue + 8 →
    marie_paper = 6 * allison_paper →
    marie_paper = 30 →
    allison_glue + allison_paper = 28 →
    marie_glue = 15 := by
  sorry

end NUMINAMATH_CALUDE_marie_glue_sticks_l1786_178612


namespace NUMINAMATH_CALUDE_angle_phi_value_l1786_178651

-- Define the problem statement
theorem angle_phi_value (φ : Real) (h1 : 0 < φ ∧ φ < π / 2) 
  (h2 : Real.sqrt 2 * Real.sin (20 * π / 180) = Real.cos φ - Real.sin φ) : 
  φ = 25 * π / 180 := by
  sorry

#check angle_phi_value

end NUMINAMATH_CALUDE_angle_phi_value_l1786_178651


namespace NUMINAMATH_CALUDE_point_on_line_l1786_178646

/-- A point is on a line if it satisfies the equation of the line formed by two other points. -/
def is_on_line (x₁ y₁ x₂ y₂ x y : ℚ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

/-- The point (5, 56/3) is on the line formed by (8, 16) and (2, 0). -/
theorem point_on_line : is_on_line 8 16 2 0 5 (56/3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1786_178646


namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l1786_178687

theorem three_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l1786_178687


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l1786_178622

/-- Given a point P on the terminal side of angle α with coordinates 
    (sin(2π/3), cos(2π/3)), prove that the smallest positive value of α is 11π/6 -/
theorem smallest_positive_angle (α : Real) : 
  (∃ (P : Real × Real), P.1 = Real.sin (2 * Real.pi / 3) ∧ 
                         P.2 = Real.cos (2 * Real.pi / 3) ∧ 
                         P ∈ {(x, y) | x = Real.sin α ∧ y = Real.cos α}) →
  (∀ β : Real, β > 0 ∧ β < α → β ≥ 11 * Real.pi / 6) ∧ 
  α = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l1786_178622


namespace NUMINAMATH_CALUDE_fish_lives_12_years_l1786_178697

/-- The lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a dog in years -/
def dog_lifespan : ℝ := 4 * hamster_lifespan

/-- The lifespan of a well-cared fish in years -/
def fish_lifespan : ℝ := dog_lifespan + 2

/-- Theorem stating that the lifespan of a well-cared fish is 12 years -/
theorem fish_lives_12_years : fish_lifespan = 12 := by sorry

end NUMINAMATH_CALUDE_fish_lives_12_years_l1786_178697


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1786_178669

def f (x : ℝ) : ℝ := -x^2 + x + 2

theorem quadratic_function_range (a : ℝ) :
  (∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a + 3 → f x₁ < f x₂) ∧
  (∃ x, a ≤ x ∧ x ≤ a + 3 ∧ f x = -4) →
  -5 ≤ a ∧ a ≤ -5/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1786_178669


namespace NUMINAMATH_CALUDE_obtuse_triangle_condition_l1786_178654

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Add triangle inequality constraints
  hpos_a : 0 < a
  hpos_b : 0 < b
  hpos_c : 0 < c
  hab : a + b > c
  hbc : b + c > a
  hca : c + a > b

-- Define what it means for a triangle to be obtuse
def is_obtuse (t : Triangle) : Prop :=
  t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2

-- State the theorem
theorem obtuse_triangle_condition (t : Triangle) :
  (t.a^2 + t.b^2 < t.c^2 → is_obtuse t) ∧
  ∃ (t' : Triangle), is_obtuse t' ∧ t'.a^2 + t'.b^2 ≥ t'.c^2 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_condition_l1786_178654


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l1786_178699

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ is_two_digit p ∧ (p ∣ binomial_coefficient 300 150) ∧
  ∀ (q : ℕ), Nat.Prime q → is_two_digit q → (q ∣ binomial_coefficient 300 150) → q ≤ p :=
by
  use 89
  sorry

#check largest_two_digit_prime_factor

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l1786_178699


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1786_178645

theorem quadratic_root_difference :
  let a : ℝ := 3 + 2 * Real.sqrt 2
  let b : ℝ := 5 + Real.sqrt 2
  let c : ℝ := -4
  let discriminant := b^2 - 4*a*c
  let root_difference := Real.sqrt discriminant / a
  root_difference = Real.sqrt (177 - 122 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1786_178645


namespace NUMINAMATH_CALUDE_balloon_difference_l1786_178607

theorem balloon_difference (x y : ℚ) 
  (eq1 : x = 2 * y - 3)
  (eq2 : y = x / 4 + 1) : 
  x - y = -5/2 := by sorry

end NUMINAMATH_CALUDE_balloon_difference_l1786_178607


namespace NUMINAMATH_CALUDE_angle_420_equivalent_to_60_l1786_178600

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem angle_420_equivalent_to_60 :
  same_terminal_side 420 60 :=
sorry

end NUMINAMATH_CALUDE_angle_420_equivalent_to_60_l1786_178600


namespace NUMINAMATH_CALUDE_mary_received_more_l1786_178653

/-- Calculates the profit share difference between two partners in a business --/
def profit_share_difference (mary_investment : ℚ) (harry_investment : ℚ) (total_profit : ℚ) : ℚ :=
  let equal_share := (1 / 3) * total_profit / 2
  let investment_based_profit := (2 / 3) * total_profit
  let mary_investment_share := (mary_investment / (mary_investment + harry_investment)) * investment_based_profit
  let harry_investment_share := (harry_investment / (mary_investment + harry_investment)) * investment_based_profit
  let mary_total := equal_share + mary_investment_share
  let harry_total := equal_share + harry_investment_share
  mary_total - harry_total

/-- Theorem stating that Mary received $800 more than Harry --/
theorem mary_received_more (mary_investment harry_investment total_profit : ℚ) :
  mary_investment = 700 →
  harry_investment = 300 →
  total_profit = 3000 →
  profit_share_difference mary_investment harry_investment total_profit = 800 := by
  sorry

#eval profit_share_difference 700 300 3000

end NUMINAMATH_CALUDE_mary_received_more_l1786_178653


namespace NUMINAMATH_CALUDE_olivias_wallet_problem_l1786_178665

/-- Given an initial amount of 78 dollars and a spending of 15 dollars,
    the remaining amount is 63 dollars. -/
theorem olivias_wallet_problem (initial_amount spent_amount remaining_amount : ℕ) : 
  initial_amount = 78 ∧ spent_amount = 15 → remaining_amount = initial_amount - spent_amount → remaining_amount = 63 := by
  sorry

end NUMINAMATH_CALUDE_olivias_wallet_problem_l1786_178665


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1786_178640

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 2) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1786_178640


namespace NUMINAMATH_CALUDE_sum_in_Q_l1786_178675

-- Define the sets P, Q, and M
def P : Set Int := {x | ∃ k, x = 2 * k}
def Q : Set Int := {x | ∃ k, x = 2 * k - 1}
def M : Set Int := {x | ∃ k, x = 4 * k + 1}

-- Theorem statement
theorem sum_in_Q (a b : Int) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_sum_in_Q_l1786_178675


namespace NUMINAMATH_CALUDE_only_odd_divisor_of_3_pow_n_plus_1_l1786_178617

theorem only_odd_divisor_of_3_pow_n_plus_1 :
  ∀ n : ℕ, Odd n → (n ∣ 3^n + 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_odd_divisor_of_3_pow_n_plus_1_l1786_178617


namespace NUMINAMATH_CALUDE_log_5_12_in_terms_of_m_n_l1786_178659

theorem log_5_12_in_terms_of_m_n (m n : ℝ) 
  (h1 : Real.log 2 / Real.log 10 = m) 
  (h2 : Real.log 3 / Real.log 10 = n) : 
  Real.log 12 / Real.log 5 = (2*m + n) / (1 - m) := by
  sorry

end NUMINAMATH_CALUDE_log_5_12_in_terms_of_m_n_l1786_178659


namespace NUMINAMATH_CALUDE_units_digit_sum_powers_l1786_178642

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_powers : units_digit ((35 ^ 7) + (93 ^ 45)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_powers_l1786_178642


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_l1786_178616

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Theorem for the simplified form of 2A - 3B
theorem simplify_2A_minus_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7 * x + 7 * y - 11 * x * y :=
sorry

-- Theorem for the value of 2A - 3B under given conditions
theorem value_2A_minus_3B :
  ∃ (x y : ℝ), x + y = -6/7 ∧ x * y = 1 ∧ 2 * A x y - 3 * B x y = -17 :=
sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_l1786_178616


namespace NUMINAMATH_CALUDE_distinct_values_of_triple_exponent_l1786_178604

-- Define the base number
def base : ℕ := 3

-- Define the function to calculate the number of distinct values
def distinct_values (n : ℕ) : ℕ :=
  -- The actual implementation is not provided, as we're only writing the statement
  sorry

-- Theorem statement
theorem distinct_values_of_triple_exponent :
  distinct_values base = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_of_triple_exponent_l1786_178604


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l1786_178679

/-- Represents the rates of biking, jogging, and swimming -/
structure Rates where
  bike : ℕ
  jog : ℕ
  swim : ℕ

/-- The problem statement -/
theorem rates_sum_of_squares (r : Rates) : r.bike^2 + r.jog^2 + r.swim^2 = 314 :=
  by
  have h1 : 2 * r.bike + 3 * r.jog + 4 * r.swim = 74 := by sorry
  have h2 : 4 * r.bike + 2 * r.jog + 3 * r.swim = 91 := by sorry
  sorry

#check rates_sum_of_squares

end NUMINAMATH_CALUDE_rates_sum_of_squares_l1786_178679


namespace NUMINAMATH_CALUDE_max_ratio_abcd_l1786_178684

theorem max_ratio_abcd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d ≥ 0)
  (h5 : (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)^2 = 3/8) :
  (∀ x y z w, x ≥ y ∧ y ≥ z ∧ z ≥ w ∧ w ≥ 0 ∧ 
    (x^2 + y^2 + z^2 + w^2) / (x + y + z + w)^2 = 3/8 →
    (x + z) / (y + w) ≤ (a + c) / (b + d)) ∧
  (a + c) / (b + d) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_abcd_l1786_178684


namespace NUMINAMATH_CALUDE_paper_thickness_l1786_178639

/-- Given that 400 sheets of paper are 4 cm thick, prove that 600 sheets of the same paper would be 6 cm thick. -/
theorem paper_thickness (sheets : ℕ) (thickness : ℝ) 
  (h1 : 400 * (thickness / 400) = 4) -- 400 sheets are 4 cm thick
  (h2 : sheets = 600) -- We want to prove for 600 sheets
  : sheets * (thickness / 400) = 6 := by
  sorry

end NUMINAMATH_CALUDE_paper_thickness_l1786_178639


namespace NUMINAMATH_CALUDE_complex_quadrant_l1786_178613

theorem complex_quadrant (z : ℂ) (h : (1 - I) / (z - 2) = 1 + I) : 
  0 < z.re ∧ z.im < 0 := by
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1786_178613


namespace NUMINAMATH_CALUDE_roots_sum_greater_than_twice_zero_l1786_178670

noncomputable section

open Real

def f (x : ℝ) := x * log x
def g (x : ℝ) := x / exp x
def F (x : ℝ) := f x - g x
def m (x : ℝ) := min (f x) (g x)

theorem roots_sum_greater_than_twice_zero 
  (x₀ : ℝ) 
  (h₁ : 1 < x₀ ∧ x₀ < 2) 
  (h₂ : F x₀ = 0) 
  (h₃ : ∀ x, 1 < x ∧ x < 2 ∧ F x = 0 → x = x₀)
  (x₁ x₂ : ℝ) 
  (h₄ : 1 < x₁ ∧ x₁ < x₂)
  (h₅ : ∃ n, m x₁ = n ∧ m x₂ = n)
  : x₁ + x₂ > 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_greater_than_twice_zero_l1786_178670


namespace NUMINAMATH_CALUDE_overtime_probability_l1786_178662

theorem overtime_probability (p_chen p_li p_both : ℝ) : 
  p_chen = 1/3 →
  p_li = 1/4 →
  p_both = 1/6 →
  p_both / p_li = 2/3 := by
sorry

end NUMINAMATH_CALUDE_overtime_probability_l1786_178662


namespace NUMINAMATH_CALUDE_income_savings_percentage_l1786_178696

theorem income_savings_percentage (I S : ℝ) 
  (h1 : S > 0) 
  (h2 : I > S) 
  (h3 : (I - S) + (1.35 * I - 2 * S) = 2 * (I - S)) : 
  S / I = 0.35 := by
sorry

end NUMINAMATH_CALUDE_income_savings_percentage_l1786_178696


namespace NUMINAMATH_CALUDE_charles_vowel_learning_time_l1786_178663

/-- The number of days Charles takes to learn one alphabet -/
def days_per_alphabet : ℕ := 7

/-- The number of vowels in the English alphabet -/
def number_of_vowels : ℕ := 5

/-- The total number of days Charles needs to finish learning all vowels -/
def total_days : ℕ := days_per_alphabet * number_of_vowels

theorem charles_vowel_learning_time : total_days = 35 := by
  sorry

end NUMINAMATH_CALUDE_charles_vowel_learning_time_l1786_178663


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l1786_178658

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  3 * x^2 + 5 * y^2 - 9 * x + 10 * y + 15 = 0

/-- Definition of an ellipse -/
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ) (A B : ℝ),
    A > 0 ∧ B > 0 ∧
    ∀ x y, f x y ↔ A * (x - h)^2 + B * (y - k)^2 = 1

/-- Theorem: The given conic equation represents an ellipse -/
theorem conic_is_ellipse : is_ellipse conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l1786_178658


namespace NUMINAMATH_CALUDE_crayons_per_child_l1786_178676

theorem crayons_per_child (total_children : ℕ) (total_crayons : ℕ) 
  (h1 : total_children = 7) 
  (h2 : total_crayons = 56) : 
  total_crayons / total_children = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_child_l1786_178676


namespace NUMINAMATH_CALUDE_larger_number_problem_l1786_178685

theorem larger_number_problem (x y : ℝ) : 
  5 * y = 6 * x → y - x = 10 → y = 60 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1786_178685


namespace NUMINAMATH_CALUDE_train_length_l1786_178633

/-- The length of a train that overtakes a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  overtake_time = 12 →
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1786_178633


namespace NUMINAMATH_CALUDE_multiple_of_seven_square_gt_200_lt_30_l1786_178637

theorem multiple_of_seven_square_gt_200_lt_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 7 * k)
  (h2 : x^2 > 200)
  (h3 : x < 30) :
  x = 21 ∨ x = 28 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_seven_square_gt_200_lt_30_l1786_178637


namespace NUMINAMATH_CALUDE_distribution_theorem_l1786_178620

/-- Represents the amount of money spent by each person -/
structure Spending where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents the amount of money received by Person A and Person B -/
structure Distribution where
  a : ℚ
  b : ℚ

/-- Calculates the distribution of money based on spending ratios -/
def calculateDistribution (s : Spending) (total : ℚ) : Distribution :=
  let ratio_sum := s.a + s.b
  let unit_value := total / ratio_sum
  { a := (s.a - s.b) * unit_value,
    b := (s.b - s.a) * unit_value + total }

/-- The main theorem to prove -/
theorem distribution_theorem (s : Spending) :
  s.b = 12/13 * s.a →
  s.c = 2/3 * s.b →
  let d := calculateDistribution s 9
  d.a = 6 ∧ d.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_distribution_theorem_l1786_178620


namespace NUMINAMATH_CALUDE_video_game_points_calculation_l1786_178615

/-- Calculate points earned in a video game level --/
theorem video_game_points_calculation
  (points_per_enemy : ℕ)
  (bonus_points : ℕ)
  (total_enemies : ℕ)
  (defeated_enemies : ℕ)
  (bonuses_earned : ℕ)
  (h1 : points_per_enemy = 15)
  (h2 : bonus_points = 50)
  (h3 : total_enemies = 25)
  (h4 : defeated_enemies = total_enemies - 5)
  (h5 : bonuses_earned = 2)
  : defeated_enemies * points_per_enemy + bonuses_earned * bonus_points = 400 := by
  sorry

#check video_game_points_calculation

end NUMINAMATH_CALUDE_video_game_points_calculation_l1786_178615


namespace NUMINAMATH_CALUDE_vector_subtraction_l1786_178608

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 4) → b = (1, 2) → a - 2 • b = (1, 0) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1786_178608


namespace NUMINAMATH_CALUDE_transform_quadratic_l1786_178635

/-- The original quadratic function -/
def g (x : ℝ) : ℝ := 2 * x^2 + 2

/-- The transformed function -/
def f (x : ℝ) : ℝ := 2 * (x + 3)^2 + 1

/-- Theorem stating that f is the result of transforming g -/
theorem transform_quadratic : 
  ∀ x : ℝ, f x = g (x + 3) - 1 := by sorry

end NUMINAMATH_CALUDE_transform_quadratic_l1786_178635


namespace NUMINAMATH_CALUDE_solve_shoe_price_l1786_178672

def shoe_price_problem (rebate_percentage : ℝ) (num_pairs : ℕ) (total_rebate : ℝ) : Prop :=
  let original_price := total_rebate / (rebate_percentage * num_pairs : ℝ)
  original_price = 28

theorem solve_shoe_price :
  shoe_price_problem 0.1 5 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_shoe_price_l1786_178672


namespace NUMINAMATH_CALUDE_gcd_91_49_l1786_178618

theorem gcd_91_49 : Nat.gcd 91 49 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_91_49_l1786_178618


namespace NUMINAMATH_CALUDE_zero_in_interval_l1786_178614

def f (x : ℝ) := 2*x + 3*x

theorem zero_in_interval : ∃ x ∈ Set.Ioo (-1 : ℝ) 0, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1786_178614


namespace NUMINAMATH_CALUDE_factorial_305_trailing_zeros_l1786_178691

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 305! ends with 75 zeros -/
theorem factorial_305_trailing_zeros :
  trailingZeros 305 = 75 := by
  sorry

end NUMINAMATH_CALUDE_factorial_305_trailing_zeros_l1786_178691


namespace NUMINAMATH_CALUDE_ratio_of_w_to_y_l1786_178601

theorem ratio_of_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 4 / 3)
  (hy_z : y / z = 3 / 2)
  (hz_x : z / x = 1 / 3) :
  w / y = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_w_to_y_l1786_178601


namespace NUMINAMATH_CALUDE_ab_value_l1786_178664

theorem ab_value (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 31) : a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1786_178664


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1786_178693

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 6
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (2 * Real.sqrt 6, Real.sqrt 2, 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1786_178693


namespace NUMINAMATH_CALUDE_special_square_area_l1786_178641

/-- A square with special points and segments -/
structure SpecialSquare where
  -- The side length of the square
  side : ℝ
  -- The length of BR
  br : ℝ
  -- The length of PR
  pr : ℝ
  -- Assumption that BR = 9
  br_eq : br = 9
  -- Assumption that PR = 12
  pr_eq : pr = 12
  -- Assumption that BP and CQ intersect at right angles
  right_angle : True

/-- The theorem stating that the area of the special square is 324 -/
theorem special_square_area (s : SpecialSquare) : s.side ^ 2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_special_square_area_l1786_178641


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1786_178677

theorem a_plus_b_value (a b : ℝ) (h1 : |a| = 3) (h2 : b^2 = 25) (h3 : a*b < 0) :
  a + b = 2 ∨ a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1786_178677


namespace NUMINAMATH_CALUDE_cow_count_l1786_178668

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ :=
  2 * group.ducks + 4 * group.cows

/-- The total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ :=
  group.ducks + group.cows

/-- Theorem: In a group where the total number of legs is 12 more than twice 
    the number of heads, the number of cows is 6 -/
theorem cow_count (group : AnimalGroup) 
    (h : totalLegs group = 2 * totalHeads group + 12) : 
    group.cows = 6 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_l1786_178668


namespace NUMINAMATH_CALUDE_circle_diameter_theorem_l1786_178695

theorem circle_diameter_theorem (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 16 * Real.pi → A = Real.pi * r^2 → d = 2 * r → 3 * d = 24 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_theorem_l1786_178695


namespace NUMINAMATH_CALUDE_radar_coverage_theorem_l1786_178610

-- Define constants
def num_radars : ℕ := 8
def radar_radius : ℝ := 15
def ring_width : ℝ := 18

-- Define the theorem
theorem radar_coverage_theorem :
  let center_to_radar : ℝ := 12 / Real.sin (22.5 * π / 180)
  let ring_area : ℝ := 432 * π / Real.tan (22.5 * π / 180)
  (∀ (r : ℝ), r = center_to_radar →
    (num_radars : ℝ) * (2 * radar_radius - ring_width) = 2 * π * r * Real.sin (π / num_radars)) ∧
  (∀ (A : ℝ), A = ring_area →
    A = π * ((r + ring_width / 2)^2 - (r - ring_width / 2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_radar_coverage_theorem_l1786_178610


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1786_178686

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1786_178686


namespace NUMINAMATH_CALUDE_calculate_expression_l1786_178681

theorem calculate_expression : (1 / 3 : ℚ) * 9 * 15 - 7 = 38 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l1786_178681


namespace NUMINAMATH_CALUDE_power_of_power_l1786_178650

theorem power_of_power (a : ℝ) : (a^3)^3 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1786_178650


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1786_178638

theorem purely_imaginary_complex_number (a : ℝ) :
  (a^2 - 4 : ℂ) + (a - 2 : ℂ) * Complex.I = (0 : ℂ) + (b : ℂ) * Complex.I ∧ b ≠ 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1786_178638


namespace NUMINAMATH_CALUDE_sam_spent_12_dimes_on_baseball_cards_l1786_178682

/-- The number of pennies Sam spent on ice cream -/
def ice_cream_pennies : ℕ := 2

/-- The total amount Sam spent in cents -/
def total_spent_cents : ℕ := 122

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of dimes Sam spent on baseball cards -/
def baseball_card_dimes : ℕ := 12

theorem sam_spent_12_dimes_on_baseball_cards :
  (total_spent_cents - ice_cream_pennies * penny_value) / dime_value = baseball_card_dimes := by
  sorry

end NUMINAMATH_CALUDE_sam_spent_12_dimes_on_baseball_cards_l1786_178682


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l1786_178602

theorem baseball_card_value_decrease (initial_value : ℝ) (h : initial_value > 0) :
  let value_after_first_year := initial_value * (1 - 0.2)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := initial_value - value_after_second_year
  let percent_decrease := (total_decrease / initial_value) * 100
  percent_decrease = 28 := by sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l1786_178602


namespace NUMINAMATH_CALUDE_a_travel_time_l1786_178694

-- Define the speed ratio of A to B
def speed_ratio : ℚ := 3 / 4

-- Define the time difference between A and B in hours
def time_difference : ℚ := 1 / 2

-- Theorem statement
theorem a_travel_time (t : ℚ) : 
  (t + time_difference) / t = 1 / speed_ratio → t + time_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_travel_time_l1786_178694


namespace NUMINAMATH_CALUDE_sequence_general_term_l1786_178657

theorem sequence_general_term (p : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n^2 + p*n) →
  (∃ r, a 2 * r = a 5 ∧ a 5 * r = a 10) →
  ∃ k, ∀ n, a n = 2*n + k :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1786_178657


namespace NUMINAMATH_CALUDE_rectangle_to_equilateral_triangle_l1786_178655

/-- Given a rectangle with length L and width W, and an equilateral triangle with side s,
    if both shapes have the same area A, then s = √(4LW/√3) -/
theorem rectangle_to_equilateral_triangle (L W s A : ℝ) (h1 : A = L * W) 
    (h2 : A = (s^2 * Real.sqrt 3) / 4) : s = Real.sqrt ((4 * L * W) / Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_equilateral_triangle_l1786_178655


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1786_178678

/-- Given a line with slope 5 passing through (-2, 3), prove m + b = 18 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 5 → 
  3 = 5 * (-2) + b → 
  m + b = 18 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1786_178678


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1786_178648

theorem inequality_solution_set :
  {x : ℝ | 3 * x - 4 > 2} = {x : ℝ | x > 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1786_178648


namespace NUMINAMATH_CALUDE_petes_nickels_spent_l1786_178611

theorem petes_nickels_spent (total_received : ℕ) (total_spent : ℕ) (raymonds_dimes_left : ℕ) 
  (h1 : total_received = 500)
  (h2 : total_spent = 200)
  (h3 : raymonds_dimes_left = 7) :
  (total_spent - (raymonds_dimes_left * 10)) / 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_petes_nickels_spent_l1786_178611


namespace NUMINAMATH_CALUDE_b_10_equals_64_l1786_178680

/-- Given two sequences {aₙ} and {bₙ} satisfying certain conditions, prove that b₁₀ = 64 -/
theorem b_10_equals_64 (a b : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n + a (n + 1) = b n)
  (h3 : ∀ n, a n * a (n + 1) = 2^n) :
  b 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_b_10_equals_64_l1786_178680


namespace NUMINAMATH_CALUDE_parabola_equation_l1786_178692

/-- A parabola with vertex at the origin and passing through (-4, 4) has the standard equation y² = -4x or x² = 4y -/
theorem parabola_equation (p : ℝ → ℝ → Prop) 
  (vertex_origin : p 0 0)
  (passes_through : p (-4) 4) :
  (∀ x y, p x y ↔ y^2 = -4*x) ∨ (∀ x y, p x y ↔ x^2 = 4*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1786_178692


namespace NUMINAMATH_CALUDE_number_of_subsets_l1786_178660

theorem number_of_subsets (S : Set ℕ) : 
  (∃ (B : Set ℕ), {1, 2} ⊆ B ∧ B ⊆ {1, 2, 3}) ∧ 
  (∀ (B : Set ℕ), {1, 2} ⊆ B ∧ B ⊆ {1, 2, 3} → B = {1, 2} ∨ B = {1, 2, 3}) :=
by sorry

end NUMINAMATH_CALUDE_number_of_subsets_l1786_178660


namespace NUMINAMATH_CALUDE_eight_person_handshakes_l1786_178647

/-- The number of handshakes in a group where each person shakes hands with every other person once -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 8 people, where each person shakes hands exactly once with every other person, the total number of handshakes is 28 -/
theorem eight_person_handshakes : num_handshakes 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_person_handshakes_l1786_178647


namespace NUMINAMATH_CALUDE_collinear_vectors_l1786_178634

/-- Given vectors a and b, if 2a - b is collinear with b, then n = 9 -/
theorem collinear_vectors (a b : ℝ × ℝ) (n : ℝ) 
  (ha : a = (1, 3))
  (hb : b = (3, n))
  (hcol : ∃ (k : ℝ), 2 • a - b = k • b) :
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_l1786_178634


namespace NUMINAMATH_CALUDE_distance_inequality_l1786_178683

-- Define the types for planes, lines, and points
variable (Plane Line Point : Type)

-- Define the distance function
variable (distance : Point → Point → ℝ)
variable (distance_point_line : Point → Line → ℝ)
variable (distance_line_line : Line → Line → ℝ)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the containment relations
variable (line_in_plane : Line → Plane → Prop)
variable (point_on_line : Point → Line → Prop)

-- Define the specific objects in the problem
variable (α β : Plane) (m n : Line) (A B : Point)

-- Define the distances
variable (a b c : ℝ)

theorem distance_inequality 
  (h_parallel : parallel α β)
  (h_m_in_α : line_in_plane m α)
  (h_n_in_β : line_in_plane n β)
  (h_A_on_m : point_on_line A m)
  (h_B_on_n : point_on_line B n)
  (h_a_def : a = distance A B)
  (h_b_def : b = distance_point_line A n)
  (h_c_def : c = distance_line_line m n) :
  c ≤ b ∧ b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_distance_inequality_l1786_178683


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1786_178625

theorem quadratic_inequality_solution (n : ℤ) : 
  n^2 - 13*n + 36 < 0 ↔ n ∈ ({5, 6, 7, 8} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1786_178625


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1786_178661

/-- Given a trapezoid ABCD where the ratio of the area of triangle ABC to the area of triangle ADC
    is 7:3, and AB + CD = 280, prove that AB = 196. -/
theorem trapezoid_segment_length (AB CD : ℝ) (h : ℝ) : 
  (AB * h / 2) / (CD * h / 2) = 7 / 3 →
  AB + CD = 280 →
  AB = 196 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l1786_178661


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1786_178623

/-- The surface area of a rectangular solid with edge lengths a, b, and c -/
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

/-- The edge lengths of the rectangular solid are prime numbers -/
axiom prime_3 : Nat.Prime 3
axiom prime_5 : Nat.Prime 5
axiom prime_17 : Nat.Prime 17

/-- The edge lengths are different -/
axiom different_edges : 3 ≠ 5 ∧ 3 ≠ 17 ∧ 5 ≠ 17

theorem rectangular_solid_surface_area :
  surface_area 3 5 17 = 302 := by sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1786_178623


namespace NUMINAMATH_CALUDE_smallest_n_property_l1786_178666

/-- The smallest positive integer N such that N and N^2 end in the same three-digit sequence abc in base 10, where a is not zero -/
def smallest_n : ℕ := 876

theorem smallest_n_property : 
  ∀ n : ℕ, n > 0 → 
  (n % 1000 = smallest_n % 1000 ∧ n^2 % 1000 = smallest_n % 1000 ∧ (smallest_n % 1000) ≥ 100) → 
  n ≥ smallest_n := by
  sorry

#eval smallest_n

end NUMINAMATH_CALUDE_smallest_n_property_l1786_178666


namespace NUMINAMATH_CALUDE_fruit_punch_water_quarts_l1786_178603

theorem fruit_punch_water_quarts 
  (water_parts juice_parts : ℕ) 
  (total_gallons : ℚ) 
  (quarts_per_gallon : ℕ) : 
  water_parts = 5 → 
  juice_parts = 2 → 
  total_gallons = 3 → 
  quarts_per_gallon = 4 → 
  (water_parts : ℚ) * total_gallons * quarts_per_gallon / (water_parts + juice_parts) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fruit_punch_water_quarts_l1786_178603


namespace NUMINAMATH_CALUDE_sine_function_monotonicity_l1786_178609

theorem sine_function_monotonicity (ω : ℝ) (h1 : ω > 0) : 
  (∀ x ∈ Set.Icc (-π/3) (π/4), 
    ∀ y ∈ Set.Icc (-π/3) (π/4), 
    x < y → 2 * Real.sin (ω * x) < 2 * Real.sin (ω * y)) 
  → 0 < ω ∧ ω ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_sine_function_monotonicity_l1786_178609


namespace NUMINAMATH_CALUDE_salesman_profit_l1786_178621

/-- Calculates the profit of a salesman selling backpacks --/
theorem salesman_profit : 
  let total_backpacks : ℕ := 48
  let cost_per_case : ℕ := 576
  let first_batch_sold : ℕ := 17
  let first_batch_price : ℕ := 18
  let second_batch_sold : ℕ := 10
  let second_batch_price : ℕ := 25
  let remaining_price : ℕ := 22
  
  let total_sales := 
    first_batch_sold * first_batch_price + 
    second_batch_sold * second_batch_price + 
    (total_backpacks - first_batch_sold - second_batch_sold) * remaining_price
  
  let profit := total_sales - cost_per_case
  
  profit = 442 := by sorry

end NUMINAMATH_CALUDE_salesman_profit_l1786_178621


namespace NUMINAMATH_CALUDE_convex_polyhedron_three_equal_edges_l1786_178636

/-- Represents an edge of a polyhedron --/
structure Edge :=
  (length : ℝ)

/-- Represents a vertex of a polyhedron --/
structure Vertex :=
  (edges : Fin 3 → Edge)

/-- Represents a convex polyhedron --/
structure ConvexPolyhedron :=
  (vertices : Set Vertex)
  (convex : Bool)
  (edge_equality : ∀ v : Vertex, v ∈ vertices → ∃ (i j : Fin 3), i ≠ j ∧ (v.edges i).length = (v.edges j).length)

/-- The main theorem: if a convex polyhedron satisfies the given conditions, it has at least three equal edges --/
theorem convex_polyhedron_three_equal_edges (P : ConvexPolyhedron) : 
  ∃ (e₁ e₂ e₃ : Edge), e₁ ≠ e₂ ∧ e₂ ≠ e₃ ∧ e₁ ≠ e₃ ∧ e₁.length = e₂.length ∧ e₂.length = e₃.length :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_three_equal_edges_l1786_178636


namespace NUMINAMATH_CALUDE_parabola_c_value_l1786_178671

/-- A parabola passing through two points with equal y-coordinates -/
structure Parabola where
  b : ℝ
  c : ℝ
  pass_through_minus_one : 2 = 1 + (-b) + c
  pass_through_three : 2 = 9 + 3*b + c

/-- The value of c for a parabola passing through (-1, 2) and (3, 2) is -1 -/
theorem parabola_c_value (p : Parabola) : p.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1786_178671


namespace NUMINAMATH_CALUDE_function_increasing_interval_implies_b_bound_l1786_178690

/-- Given a function f(x) = e^x(x^2 - bx) where b is a real number,
    if f(x) has an increasing interval in [1/2, 2],
    then b < 8/3 -/
theorem function_increasing_interval_implies_b_bound 
  (b : ℝ) 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = Real.exp x * (x^2 - b*x)) 
  (h_increasing : ∃ (a c : ℝ), 1/2 ≤ a ∧ c ≤ 2 ∧ StrictMonoOn f (Set.Icc a c)) : 
  b < 8/3 :=
sorry

end NUMINAMATH_CALUDE_function_increasing_interval_implies_b_bound_l1786_178690


namespace NUMINAMATH_CALUDE_tournament_winner_percentage_l1786_178667

theorem tournament_winner_percentage (n : ℕ) (total_games : ℕ) 
  (top_player_advantage : ℝ) (least_successful_percentage : ℝ) 
  (remaining_players_percentage : ℝ) :
  n = 8 →
  total_games = 560 →
  top_player_advantage = 0.15 →
  least_successful_percentage = 0.08 →
  remaining_players_percentage = 0.35 →
  ∃ (top_player_percentage : ℝ),
    top_player_percentage = 0.395 ∧
    top_player_percentage = 
      (1 - (2 * least_successful_percentage + remaining_players_percentage)) / 2 + 
      top_player_advantage :=
by sorry

end NUMINAMATH_CALUDE_tournament_winner_percentage_l1786_178667


namespace NUMINAMATH_CALUDE_solution_for_all_polynomials_l1786_178649

/-- A polynomial of degree 3 in x and y -/
def q (b₁ b₂ b₄ b₇ b₈ : ℝ) (x y : ℝ) : ℝ :=
  b₁ * x * (1 - x^2) + b₂ * y * (1 - y^2) + b₄ * (x * y - x^2 * y) + b₇ * x^2 * y + b₈ * x * y^2

/-- The theorem stating that (√(3/2), √(3/2)) is a solution for all such polynomials -/
theorem solution_for_all_polynomials (b₁ b₂ b₄ b₇ b₈ : ℝ) :
  let q := q b₁ b₂ b₄ b₇ b₈
  (q 0 0 = 0) →
  (q 1 0 = 0) →
  (q (-1) 0 = 0) →
  (q 0 1 = 0) →
  (q 0 (-1) = 0) →
  (q 1 1 = 0) →
  (q (-1) (-1) = 0) →
  (q 2 2 = 0) →
  (deriv (fun x => q x 1) 1 = 0) →
  (deriv (fun y => q 1 y) 1 = 0) →
  q (Real.sqrt (3/2)) (Real.sqrt (3/2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_all_polynomials_l1786_178649


namespace NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l1786_178674

theorem largest_x_absolute_value_equation : 
  (∃ x : ℝ, |5*x - 3| = 28) → 
  (∃ max_x : ℝ, |5*max_x - 3| = 28 ∧ ∀ y : ℝ, |5*y - 3| = 28 → y ≤ max_x) → 
  (∃ x : ℝ, |5*x - 3| = 28 ∧ ∀ y : ℝ, |5*y - 3| = 28 → y ≤ 31/5) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l1786_178674


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1786_178631

theorem rectangle_area_diagonal (l w d : ℝ) (h_ratio : l / w = 5 / 2) (h_diag : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1786_178631


namespace NUMINAMATH_CALUDE_expression_value_l1786_178619

theorem expression_value (x y : ℤ) (hx : x = -6) (hy : y = -3) :
  (x - y)^2 - x*y = -9 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1786_178619


namespace NUMINAMATH_CALUDE_polynomial_identity_l1786_178627

theorem polynomial_identity (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃)) :
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1786_178627


namespace NUMINAMATH_CALUDE_power_division_rule_l1786_178644

theorem power_division_rule (x : ℝ) : x^10 / x^2 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1786_178644


namespace NUMINAMATH_CALUDE_ear_muffs_december_l1786_178606

theorem ear_muffs_december (before_december : ℕ) (total : ℕ) (during_december : ℕ) : 
  before_december = 1346 →
  total = 7790 →
  during_december = total - before_december →
  during_december = 6444 := by
sorry

end NUMINAMATH_CALUDE_ear_muffs_december_l1786_178606


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l1786_178629

theorem continued_fraction_solution :
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) → x = (3 + Real.sqrt 69) / 2 :=
by sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l1786_178629


namespace NUMINAMATH_CALUDE_quadratic_distinct_rational_roots_l1786_178656

theorem quadratic_distinct_rational_roots (a b c : ℚ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hsum : a + b + c = 0) : 
  ∃ (x y : ℚ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_rational_roots_l1786_178656


namespace NUMINAMATH_CALUDE_coffee_shop_tables_l1786_178643

def base7_to_base10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

theorem coffee_shop_tables (chairs_base7 : Nat) (people_per_table : Nat) : 
  chairs_base7 = 321 ∧ people_per_table = 3 → 
  (base7_to_base10 chairs_base7) / people_per_table = 54 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_tables_l1786_178643


namespace NUMINAMATH_CALUDE_television_regular_price_l1786_178630

theorem television_regular_price (sale_price : ℝ) (discount_rate : ℝ) (regular_price : ℝ) :
  sale_price = regular_price * (1 - discount_rate) →
  discount_rate = 0.2 →
  sale_price = 480 →
  regular_price = 600 := by
sorry

end NUMINAMATH_CALUDE_television_regular_price_l1786_178630


namespace NUMINAMATH_CALUDE_symmetry_sum_theorem_l1786_178673

/-- Properties of a regular 25-gon -/
structure RegularPolygon25 where
  /-- Number of lines of symmetry -/
  L : ℕ
  /-- Smallest positive angle for rotational symmetry in degrees -/
  R : ℝ
  /-- The polygon has 25 sides -/
  sides_eq : L = 25
  /-- The smallest rotational symmetry angle is 360/25 degrees -/
  angle_eq : R = 360 / 25

/-- Theorem about the sum of symmetry lines and half the rotational angle -/
theorem symmetry_sum_theorem (p : RegularPolygon25) :
  p.L + p.R / 2 = 32.2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_theorem_l1786_178673


namespace NUMINAMATH_CALUDE_k_range_theorem_l1786_178689

def f (x : ℝ) : ℝ := x * abs x

theorem k_range_theorem (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ici 1 ∧ f (x - 2*k) < k) → k ∈ Set.Ioi (1/4) :=
by sorry

end NUMINAMATH_CALUDE_k_range_theorem_l1786_178689


namespace NUMINAMATH_CALUDE_total_fruits_is_fifteen_l1786_178626

-- Define the three types of fruit
inductive FruitType
| A
| B
| C

-- Define a function that returns the quantity of each fruit type
def fruitQuantity (t : FruitType) : Nat :=
  match t with
  | FruitType.A => 5
  | FruitType.B => 6
  | FruitType.C => 4

-- Theorem: The total number of fruits is 15
theorem total_fruits_is_fifteen :
  (fruitQuantity FruitType.A) + (fruitQuantity FruitType.B) + (fruitQuantity FruitType.C) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fruits_is_fifteen_l1786_178626


namespace NUMINAMATH_CALUDE_class_enrollment_l1786_178652

theorem class_enrollment (q1_correct q2_correct both_correct not_taken : ℕ) 
  (h1 : q1_correct = 25)
  (h2 : q2_correct = 22)
  (h3 : not_taken = 5)
  (h4 : both_correct = 22) :
  q1_correct + q2_correct - both_correct + not_taken = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_class_enrollment_l1786_178652


namespace NUMINAMATH_CALUDE_recipe_salt_amount_l1786_178688

def recipe_salt (total_flour sugar flour_added : ℕ) : ℕ :=
  let remaining_flour := total_flour - flour_added
  remaining_flour - 3

theorem recipe_salt_amount :
  recipe_salt 12 14 2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_recipe_salt_amount_l1786_178688


namespace NUMINAMATH_CALUDE_triangle_area_prime_l1786_178624

/-- The area of a triangle formed by the line y = 10x - a and the coordinate axes -/
def triangleArea (a : ℤ) : ℚ := a^2 / 20

/-- Predicate to check if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

theorem triangle_area_prime :
  ∀ a : ℤ,
  (∃ n : ℕ, (triangleArea a).num = n ∧ (triangleArea a).den = 1 ∧ isPrime n) →
  triangleArea a = 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_prime_l1786_178624


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1786_178632

def P : Set ℝ := {x | x^2 - x = 0}
def Q : Set ℝ := {x | x^2 + x = 0}

theorem intersection_P_Q : P ∩ Q = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1786_178632


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l1786_178698

theorem average_of_four_numbers (p q r s : ℝ) 
  (h : (5 / 4) * (p + q + r + s) = 20) : 
  (p + q + r + s) / 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l1786_178698


namespace NUMINAMATH_CALUDE_quadratic_radical_sum_l1786_178605

/-- 
Given that √(3b-1) and ∜(7-b) are of the same type of quadratic radical,
where ∜ represents the (a-1)th root, prove that a + b = 5.
-/
theorem quadratic_radical_sum (a b : ℝ) : 
  (a - 1 = 2) → (3*b - 1 = 7 - b) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_sum_l1786_178605
