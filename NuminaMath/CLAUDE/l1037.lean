import Mathlib

namespace cube_of_difference_l1037_103779

theorem cube_of_difference (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a^2 + b^2 = 98) : 
  (a - b)^3 = 512 := by
  sorry

end cube_of_difference_l1037_103779


namespace symmetry_implies_ratio_l1037_103777

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℚ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if points A(m+4, -1) and B(1, n-3) are symmetric with respect to the origin,
    then m/n = -5/4 -/
theorem symmetry_implies_ratio (m n : ℚ) :
  symmetric_wrt_origin (m + 4) (-1) 1 (n - 3) →
  m / n = -5 / 4 :=
by sorry

end symmetry_implies_ratio_l1037_103777


namespace valid_regression_equation_l1037_103769

-- Define the linear regression equation
def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Define the theorem
theorem valid_regression_equation :
  -- Conditions
  ∀ (x_mean y_mean : ℝ),
  x_mean = 3 →
  y_mean = 3.5 →
  -- The regression equation
  ∃ (a b : ℝ),
  -- Positive correlation
  a > 0 ∧
  -- Equation passes through (x_mean, y_mean)
  linear_regression a b x_mean = y_mean ∧
  -- Specific coefficients
  a = 0.4 ∧
  b = 2.3 :=
by
  sorry


end valid_regression_equation_l1037_103769


namespace money_distribution_l1037_103751

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 400)
  (AC_sum : A + C = 300)
  (BC_sum : B + C = 150) : 
  C = 50 := by
sorry

end money_distribution_l1037_103751


namespace max_value_of_complex_expression_l1037_103783

theorem max_value_of_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  Complex.abs ((w - 2)^2 * (w + 2)) ≤ 24 ∧
  ∃ w₀ : ℂ, Complex.abs w₀ = 2 ∧ Complex.abs ((w₀ - 2)^2 * (w₀ + 2)) = 24 :=
by sorry

end max_value_of_complex_expression_l1037_103783


namespace sin_two_alpha_value_l1037_103720

theorem sin_two_alpha_value (α : Real) 
  (h1 : 0 < α ∧ α < π) 
  (h2 : (1/2) * Real.cos (2*α) = Real.sin (π/4 + α)) : 
  Real.sin (2*α) = -1 := by
  sorry

end sin_two_alpha_value_l1037_103720


namespace expression_evaluation_l1037_103735

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 11)
  (h2 : b = a + 3)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 72 / 35 := by
sorry

end expression_evaluation_l1037_103735


namespace max_value_of_f_l1037_103759

/-- The quadratic function f(x) = -2x^2 + 9 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 9

/-- Theorem: The maximum value of f(x) = -2x^2 + 9 is 9 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l1037_103759


namespace intersection_A_B_range_of_a_l1037_103706

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for the range of a when B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end intersection_A_B_range_of_a_l1037_103706


namespace xyz_value_l1037_103714

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 14 / 3 := by
sorry

end xyz_value_l1037_103714


namespace pet_store_cages_l1037_103757

def cages_used (total_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : ℕ :=
  let remaining_puppies := total_puppies - sold_puppies
  (remaining_puppies / puppies_per_cage) + if remaining_puppies % puppies_per_cage > 0 then 1 else 0

theorem pet_store_cages :
  cages_used 1700 621 26 = 42 := by
  sorry

end pet_store_cages_l1037_103757


namespace parabola_tangent_intersection_l1037_103796

/-- Proves that for a parabola y^2 = 2px, the intersection point of two tangent lines
    has a y-coordinate equal to the average of the y-coordinates of the tangent points. -/
theorem parabola_tangent_intersection
  (p : ℝ) (x₁ y₁ x₂ y₂ x y : ℝ)
  (h_parabola₁ : y₁^2 = 2*p*x₁)
  (h_parabola₂ : y₂^2 = 2*p*x₂)
  (h_tangent₁ : y*y₁ = p*(x + x₁))
  (h_tangent₂ : y*y₂ = p*(x + x₂))
  (h_distinct : y₁ ≠ y₂) :
  y = (y₁ + y₂) / 2 :=
by sorry

end parabola_tangent_intersection_l1037_103796


namespace sufficient_but_not_necessary_l1037_103744

theorem sufficient_but_not_necessary (p q : Prop) 
  (h : (¬p → ¬q) ∧ ¬(¬q → ¬p)) : 
  (p → q) ∧ ¬(q → p) := by
  sorry

end sufficient_but_not_necessary_l1037_103744


namespace data_analysis_l1037_103764

def dataset : List ℕ := [10, 8, 6, 9, 8, 7, 8]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

theorem data_analysis (l : List ℕ) (h : l = dataset) : 
  (mode l = 8) ∧ 
  (median l = 8) ∧ 
  (mean l = 8) ∧ 
  (variance l ≠ 8) := by sorry

end data_analysis_l1037_103764


namespace complex_real_condition_l1037_103771

theorem complex_real_condition (a : ℝ) : 
  (((1 : ℂ) + Complex.I) ^ 2 - a / Complex.I).im = 0 → a = -2 := by
  sorry

end complex_real_condition_l1037_103771


namespace nelly_outbid_joe_l1037_103705

def joes_bid : ℕ := 160000
def nellys_bid : ℕ := 482000

theorem nelly_outbid_joe : nellys_bid - 3 * joes_bid = 2000 := by
  sorry

end nelly_outbid_joe_l1037_103705


namespace aku_birthday_friends_l1037_103704

/-- Given the conditions of Aku's birthday party, prove the number of friends invited. -/
theorem aku_birthday_friends (packages : Nat) (cookies_per_package : Nat) (cookies_per_child : Nat) :
  packages = 3 →
  cookies_per_package = 25 →
  cookies_per_child = 15 →
  (packages * cookies_per_package) / cookies_per_child - 1 = 4 := by
  sorry

#eval (3 * 25) / 15 - 1  -- Expected output: 4

end aku_birthday_friends_l1037_103704


namespace board_coverage_uncoverable_boards_l1037_103747

/-- Represents a rectangular board, possibly with one square removed -/
structure Board where
  rows : Nat
  cols : Nat
  removed : Bool

/-- Calculates the total number of squares on a board -/
def Board.totalSquares (b : Board) : Nat :=
  b.rows * b.cols - if b.removed then 1 else 0

/-- Predicate for whether a board can be covered by dominoes -/
def canBeCovered (b : Board) : Prop :=
  b.totalSquares % 2 = 0

/-- Main theorem: A board can be covered iff its total squares is even -/
theorem board_coverage (b : Board) :
  canBeCovered b ↔ b.totalSquares % 2 = 0 := by sorry

/-- Specific boards from the problem -/
def board_3x4 : Board := { rows := 3, cols := 4, removed := false }
def board_3x5 : Board := { rows := 3, cols := 5, removed := false }
def board_4x4_removed : Board := { rows := 4, cols := 4, removed := true }
def board_5x5 : Board := { rows := 5, cols := 5, removed := false }
def board_6x3 : Board := { rows := 6, cols := 3, removed := false }

/-- Theorem about which boards cannot be covered -/
theorem uncoverable_boards :
  ¬(canBeCovered board_3x5) ∧
  ¬(canBeCovered board_4x4_removed) ∧
  ¬(canBeCovered board_5x5) ∧
  (canBeCovered board_3x4) ∧
  (canBeCovered board_6x3) := by sorry

end board_coverage_uncoverable_boards_l1037_103747


namespace mork_tax_rate_l1037_103750

/-- Represents the tax rates and incomes of Mork and Mindy -/
structure TaxData where
  mork_income : ℝ
  mork_tax_rate : ℝ
  mindy_tax_rate : ℝ
  combined_tax_rate : ℝ

/-- The conditions of the problem -/
def tax_conditions (data : TaxData) : Prop :=
  data.mork_income > 0 ∧
  data.mindy_tax_rate = 0.25 ∧
  data.combined_tax_rate = 0.29

/-- The theorem stating Mork's tax rate given the conditions -/
theorem mork_tax_rate (data : TaxData) :
  tax_conditions data →
  data.mork_tax_rate = 0.45 := by
  sorry


end mork_tax_rate_l1037_103750


namespace right_tangential_trapezoid_shorter_leg_l1037_103760

/-- In a right tangential trapezoid, the shorter leg equals 2ac/(a+c) where a and c are the lengths of the bases. -/
theorem right_tangential_trapezoid_shorter_leg
  (a c d : ℝ)
  (h_positive : a > 0 ∧ c > 0 ∧ d > 0)
  (h_right_tangential : d^2 + (a - c)^2 = (a + c - d)^2)
  (h_shorter_leg : d ≤ a + c - d) :
  d = 2 * a * c / (a + c) := by
sorry

end right_tangential_trapezoid_shorter_leg_l1037_103760


namespace ellipse_y_equation_ellipse_x_equation_l1037_103718

/-- An ellipse with foci on the y-axis -/
structure EllipseY where
  c : ℝ
  e : ℝ

/-- An ellipse passing through a point on the x-axis -/
structure EllipseX where
  x : ℝ
  e : ℝ

/-- Standard equation of an ellipse -/
def standardEquation (a b : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_y_equation (E : EllipseY) (h1 : E.c = 6) (h2 : E.e = 2/3) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ standardEquation a b = standardEquation 9 (Real.sqrt 45) :=
sorry

theorem ellipse_x_equation (E : EllipseX) (h1 : E.x = 2) (h2 : E.e = Real.sqrt 3 / 2) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ standardEquation a b = standardEquation 2 1) ∨
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ standardEquation a b = standardEquation 4 2) :=
sorry

end ellipse_y_equation_ellipse_x_equation_l1037_103718


namespace a_minus_b_equals_negative_nine_l1037_103772

theorem a_minus_b_equals_negative_nine
  (a b : ℝ)
  (h : |a + 5| + Real.sqrt (2 * b - 8) = 0) :
  a - b = -9 := by
sorry

end a_minus_b_equals_negative_nine_l1037_103772


namespace obtuse_triangle_side_range_l1037_103746

theorem obtuse_triangle_side_range (a : ℝ) : 
  (∃ (θ : ℝ), 
    -- Triangle inequality
    a > 1 ∧ 
    -- Obtuse triangle condition
    π/2 < θ ∧ 
    -- Largest angle doesn't exceed 120°
    θ ≤ 2*π/3 ∧ 
    -- Cosine law for the largest angle
    Real.cos θ = (a^2 + (a+1)^2 - (a+2)^2) / (2*a*(a+1))) 
  ↔ 
  (3/2 ≤ a ∧ a < 3) :=
by sorry

end obtuse_triangle_side_range_l1037_103746


namespace gcd_5280_2155_l1037_103738

theorem gcd_5280_2155 : Nat.gcd 5280 2155 = 5 := by
  sorry

end gcd_5280_2155_l1037_103738


namespace log_sum_equals_two_l1037_103711

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end log_sum_equals_two_l1037_103711


namespace complex_cube_root_l1037_103765

theorem complex_cube_root (a b : ℕ+) (h : (↑a + Complex.I * ↑b) ^ 3 = 2 + 11 * Complex.I) :
  ↑a + Complex.I * ↑b = 2 + Complex.I := by
  sorry

end complex_cube_root_l1037_103765


namespace goose_survival_rate_l1037_103753

theorem goose_survival_rate (total_eggs : ℝ) (hatched_fraction : ℝ) (first_year_survival_fraction : ℝ) (first_year_survivors : ℕ) : 
  total_eggs = 550 →
  hatched_fraction = 2/3 →
  first_year_survival_fraction = 2/5 →
  first_year_survivors = 110 →
  ∃ (first_month_survival_fraction : ℝ),
    first_month_survival_fraction * hatched_fraction * first_year_survival_fraction * total_eggs = first_year_survivors ∧
    first_month_survival_fraction = 3/4 := by
  sorry

end goose_survival_rate_l1037_103753


namespace rogers_wife_is_anne_l1037_103752

-- Define the set of people
inductive Person : Type
  | Henry | Peter | Louis | Roger | Elizabeth | Jeanne | Mary | Anne

-- Define the relationship of being married
def married : Person → Person → Prop := sorry

-- Define the action of dancing
def dancing : Person → Prop := sorry

-- Define the action of playing an instrument
def playing : Person → String → Prop := sorry

theorem rogers_wife_is_anne :
  -- Conditions
  (∀ p : Person, ∃! q : Person, married p q) →
  (∃ p : Person, married Person.Henry p ∧ dancing p ∧ 
    ∃ q : Person, married q Person.Elizabeth ∧ dancing q) →
  (¬ dancing Person.Roger) →
  (¬ dancing Person.Anne) →
  (playing Person.Peter "trumpet") →
  (playing Person.Mary "piano") →
  (¬ married Person.Anne Person.Peter) →
  -- Conclusion
  married Person.Roger Person.Anne :=
by sorry

end rogers_wife_is_anne_l1037_103752


namespace share_distribution_l1037_103767

theorem share_distribution (total : ℝ) (y_share : ℝ) (x_to_y_ratio : ℝ) :
  total = 273 →
  y_share = 63 →
  x_to_y_ratio = 0.45 →
  ∃ (x_share z_share : ℝ),
    y_share = x_to_y_ratio * x_share ∧
    total = x_share + y_share + z_share ∧
    z_share / x_share = 0.5 := by
  sorry

end share_distribution_l1037_103767


namespace basketball_tournament_l1037_103739

/-- Number of classes in the tournament -/
def num_classes : ℕ := 10

/-- Total number of matches in the tournament -/
def total_matches : ℕ := 45

/-- Points earned for winning a game -/
def win_points : ℕ := 2

/-- Points earned for losing a game -/
def lose_points : ℕ := 1

/-- Minimum points target for a class -/
def min_points : ℕ := 14

/-- Theorem stating the number of classes and minimum wins required -/
theorem basketball_tournament :
  (num_classes * (num_classes - 1)) / 2 = total_matches ∧
  ∃ (min_wins : ℕ), 
    min_wins * win_points + (num_classes - 1 - min_wins) * lose_points ≥ min_points ∧
    ∀ (wins : ℕ), wins < min_wins → 
      wins * win_points + (num_classes - 1 - wins) * lose_points < min_points :=
by sorry

end basketball_tournament_l1037_103739


namespace six_digit_divisible_by_nine_l1037_103770

theorem six_digit_divisible_by_nine :
  ∃! d : ℕ, d < 10 ∧ (135790 + d) % 9 = 0 :=
by sorry

end six_digit_divisible_by_nine_l1037_103770


namespace method1_is_optimal_l1037_103784

/-- Represents the three methods of division available to the economist. -/
inductive DivisionMethod
  | method1
  | method2
  | method3

/-- Represents a division of coins. -/
structure Division where
  total : ℕ
  part1 : ℕ
  part2 : ℕ
  part3 : ℕ
  part4 : ℕ

/-- The coin division problem. -/
def CoinDivisionProblem (n : ℕ) (method : DivisionMethod) : Prop :=
  -- The total number of coins is odd and greater than 4
  n % 2 = 1 ∧ n > 4 ∧
  -- There exists a valid division of coins
  ∃ (div : Division),
    -- The total number of coins is correct
    div.total = n ∧
    -- Each part has at least one coin
    div.part1 ≥ 1 ∧ div.part2 ≥ 1 ∧ div.part3 ≥ 1 ∧ div.part4 ≥ 1 ∧
    -- The sum of all parts equals the total
    div.part1 + div.part2 + div.part3 + div.part4 = n ∧
    -- The lawyer's initial division results in two parts with at least 2 coins each
    div.part1 + div.part2 ≥ 2 ∧ div.part3 + div.part4 ≥ 2 ∧
    -- Method 1 is the optimal strategy
    (method = DivisionMethod.method1 →
      div.part1 + div.part4 > div.part2 + div.part3 ∧
      div.part1 + div.part4 > div.part1 + div.part2 - 1)

/-- Theorem stating that Method 1 is the optimal strategy for the economist. -/
theorem method1_is_optimal (n : ℕ) :
  CoinDivisionProblem n DivisionMethod.method1 := by
  sorry


end method1_is_optimal_l1037_103784


namespace fraction_equality_l1037_103703

theorem fraction_equality : (81081 : ℝ)^4 / (27027 : ℝ)^4 = 81 := by
  sorry

end fraction_equality_l1037_103703


namespace max_popsicles_is_16_l1037_103708

/-- Represents the cost and quantity of a popsicle package -/
structure PopsiclePackage where
  cost : ℕ
  quantity : ℕ

/-- Calculates the maximum number of popsicles that can be bought with a given budget -/
def maxPopsicles (budget : ℕ) (packages : List PopsiclePackage) : ℕ := sorry

/-- The specific problem setup -/
def problemSetup : List PopsiclePackage := [
  ⟨1, 1⟩,  -- Single popsicle
  ⟨3, 3⟩,  -- 3-popsicle box
  ⟨4, 7⟩   -- 7-popsicle box
]

/-- Theorem stating that the maximum number of popsicles Pablo can buy is 16 -/
theorem max_popsicles_is_16 :
  maxPopsicles 10 problemSetup = 16 := by sorry

end max_popsicles_is_16_l1037_103708


namespace w_squared_value_l1037_103795

theorem w_squared_value (w : ℝ) (h : (w + 15)^2 = (4*w + 6)*(2*w + 3)) : w^2 = 207/7 := by
  sorry

end w_squared_value_l1037_103795


namespace min_value_theorem_l1037_103710

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → 3*x + 2*y + y/x ≥ 11 :=
by sorry

end min_value_theorem_l1037_103710


namespace surf_festival_average_l1037_103716

theorem surf_festival_average (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) :
  day1 = 1500 →
  day2 = day1 + 600 →
  day3 = (2 : ℕ) * day1 / 5 →
  (day1 + day2 + day3) / 3 = 1400 := by
  sorry

end surf_festival_average_l1037_103716


namespace function_and_composition_proof_l1037_103776

def f (x b c : ℝ) : ℝ := x^2 - b*x + c

theorem function_and_composition_proof 
  (b c : ℝ) 
  (h1 : f 1 b c = 0) 
  (h2 : f 2 b c = -3) :
  (∀ x, f x b c = x^2 - 6*x + 5) ∧ 
  (∀ x, x > -1 → f (1 / Real.sqrt (x + 1)) b c = 1 / (x + 1) - 6 / Real.sqrt (x + 1) + 5) :=
by sorry

end function_and_composition_proof_l1037_103776


namespace min_max_abs_quadratic_minus_linear_l1037_103762

/-- The minimum value of max_{0 ≤ x ≤ 2} |x^2 - 2xy| over all real y is 4√2 -/
theorem min_max_abs_quadratic_minus_linear :
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2*x*y| ≥ 4 * Real.sqrt 2) ∧
  (∃ y : ℝ, ∀ x : ℝ, 0 ≤ x → x ≤ 2 → |x^2 - 2*x*y| ≤ 4 * Real.sqrt 2) :=
by sorry

end min_max_abs_quadratic_minus_linear_l1037_103762


namespace perimeter_of_parallelogram_PSTU_l1037_103787

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  pq_eq_pr : dist P Q = dist P R
  pq_eq_15 : dist P Q = 15
  qr_eq_14 : dist Q R = 14

-- Define points S, T, U on the sides of the triangle
def S (P Q : ℝ × ℝ) : ℝ × ℝ := sorry
def T (Q R : ℝ × ℝ) : ℝ × ℝ := sorry
def U (P R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the parallelism conditions
def parallel (A B C D : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem perimeter_of_parallelogram_PSTU (P Q R : ℝ × ℝ) 
  (h : Triangle P Q R) 
  (h_st_parallel : parallel (S P Q) (T Q R) P R)
  (h_tu_parallel : parallel (T Q R) (U P R) P Q) : 
  dist P (S P Q) + dist (S P Q) (T Q R) + dist (T Q R) (U P R) + dist (U P R) P = 30 := by
  sorry


end perimeter_of_parallelogram_PSTU_l1037_103787


namespace correlation_coefficient_is_one_l1037_103743

/-- A structure representing a set of sample data points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  n_ge_2 : n ≥ 2
  x_not_all_equal : ∃ i j, i ≠ j ∧ x i ≠ x j
  points_on_line : ∀ i, y i = 3 * x i + 1

/-- The correlation coefficient of a set of sample data -/
def correlationCoefficient (data : SampleData) : ℝ :=
  sorry

/-- Theorem stating that the correlation coefficient is 1 for the given conditions -/
theorem correlation_coefficient_is_one (data : SampleData) : 
  correlationCoefficient data = 1 :=
sorry

end correlation_coefficient_is_one_l1037_103743


namespace candidates_per_state_l1037_103793

theorem candidates_per_state (candidates : ℕ) : 
  (candidates * 6 / 100 : ℚ) + 80 = candidates * 7 / 100 → candidates = 8000 := by
  sorry

end candidates_per_state_l1037_103793


namespace quadratic_discriminant_theorem_l1037_103774

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ := p.b^2 - 4*p.a*p.c

/-- A function to check if a quadratic equation has exactly one root -/
def has_one_root (a b c : ℝ) : Prop := (b^2 - 4*a*c = 0)

theorem quadratic_discriminant_theorem (p : QuadraticPolynomial) 
  (h1 : has_one_root p.a p.b (p.c + 2))
  (h2 : has_one_root p.a (p.b + 1/2) (p.c - 1)) :
  discriminant p = -1/2 := by
  sorry

end quadratic_discriminant_theorem_l1037_103774


namespace data_plan_total_cost_l1037_103700

/-- Calculates the total cost of a data plan over 6 months with special conditions -/
def data_plan_cost (regular_charge : ℚ) (promo_rate : ℚ) (extra_fee : ℚ) : ℚ :=
  let first_month := regular_charge * promo_rate
  let fourth_month := regular_charge + extra_fee
  let regular_months := 4 * regular_charge
  first_month + fourth_month + regular_months

/-- Proves that the total cost for the given conditions is $175 -/
theorem data_plan_total_cost :
  data_plan_cost 30 (1/3) 15 = 175 := by
  sorry

#eval data_plan_cost 30 (1/3) 15

end data_plan_total_cost_l1037_103700


namespace average_equation_solution_l1037_103733

theorem average_equation_solution (x : ℝ) : 
  ((2 * x + 12 + 5 * x^2 + 3 * x + 1 + 3 * x + 14) / 3 = 6 * x^2 + x - 21) ↔ 
  (x = (5 + Real.sqrt 4705) / 26 ∨ x = (5 - Real.sqrt 4705) / 26) :=
by sorry

end average_equation_solution_l1037_103733


namespace composition_of_even_is_even_l1037_103707

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end composition_of_even_is_even_l1037_103707


namespace range_of_m_correct_l1037_103731

/-- The range of m satisfying the given conditions -/
def range_of_m : Set ℝ :=
  {m | m ≥ 3 ∨ (1 < m ∧ m ≤ 2)}

/-- Condition p: x^2 + mx + 1 = 0 has two distinct negative roots -/
def condition_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
    x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Condition q: 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def condition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m_correct :
  ∀ m : ℝ, (condition_p m ∨ condition_q m) ∧ ¬(condition_p m ∧ condition_q m) ↔ m ∈ range_of_m :=
by sorry

end range_of_m_correct_l1037_103731


namespace polynomial_parity_and_divisibility_l1037_103736

theorem polynomial_parity_and_divisibility (p q : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k ↔ p % 2 = 1 ∧ q % 2 = 0) ∧
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k + 1 ↔ p % 2 = 1 ∧ q % 2 = 1) ∧
  (∀ x : ℤ, ∃ k : ℤ, x^3 + p*x + q = 3*k ↔ q % 3 = 0 ∧ p % 3 = 2) :=
by sorry

end polynomial_parity_and_divisibility_l1037_103736


namespace square_in_right_triangle_l1037_103726

/-- Given a right triangle PQR with PQ = 9, PR = 12, and right angle at P,
    if a square is fitted with one side on the hypotenuse QR and other vertices
    touching the legs of the triangle, then the length of the square's side is 3. -/
theorem square_in_right_triangle (P Q R : ℝ × ℝ) (s : ℝ) :
  let pq : ℝ := 9
  let pr : ℝ := 12
  -- P is the origin (0, 0)
  P = (0, 0) →
  -- Q is on the x-axis
  Q.2 = 0 →
  -- R is on the y-axis
  R.1 = 0 →
  -- PQ = 9
  Q.1 = pq →
  -- PR = 12
  R.2 = pr →
  -- s is positive
  s > 0 →
  -- One vertex of the square is on QR
  ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ x + y = s ∧ x^2 + y^2 = (pq - s)^2 + (pr - s)^2 →
  -- The square's side length is 3
  s = 3 :=
by sorry

end square_in_right_triangle_l1037_103726


namespace junk_items_remaining_l1037_103797

/-- Represents the distribution of items in Mason's attic -/
structure AtticItems where
  useful : ℕ
  valuable : ℕ
  junk : ℕ

/-- Calculates the total number of items in the attic -/
def AtticItems.total (items : AtticItems) : ℕ := items.useful + items.valuable + items.junk

/-- The initial distribution of items in the attic -/
def initial_distribution : AtticItems := {
  useful := 20,
  valuable := 10,
  junk := 70
}

/-- The number of useful items given away -/
def useful_items_given_away : ℕ := 4

/-- The number of valuable items sold -/
def valuable_items_sold : ℕ := 20

/-- The number of useful items remaining after giving some away -/
def remaining_useful_items : ℕ := 16

/-- Theorem stating the number of junk items remaining in the attic -/
theorem junk_items_remaining (h1 : initial_distribution.total = 100)
  (h2 : remaining_useful_items = initial_distribution.useful - useful_items_given_away)
  (h3 : valuable_items_sold > initial_distribution.valuable) :
  initial_distribution.junk - (valuable_items_sold - initial_distribution.valuable) = 60 := by
  sorry


end junk_items_remaining_l1037_103797


namespace plant_initial_length_proof_l1037_103713

/-- The daily growth rate of the plant in feet -/
def daily_growth : ℝ := 0.6875

/-- The initial length of the plant in feet -/
def initial_length : ℝ := 11

/-- The length of the plant after n days -/
def plant_length (n : ℕ) : ℝ := initial_length + n * daily_growth

theorem plant_initial_length_proof :
  (plant_length 10 = 1.3 * plant_length 4) →
  initial_length = 11 := by
  sorry

end plant_initial_length_proof_l1037_103713


namespace number_of_elements_in_set_l1037_103798

theorem number_of_elements_in_set
  (initial_average : ℚ)
  (misread_number : ℚ)
  (correct_number : ℚ)
  (correct_average : ℚ)
  (h1 : initial_average = 18)
  (h2 : misread_number = 26)
  (h3 : correct_number = 36)
  (h4 : correct_average = 19) :
  ∃ (n : ℕ), (n : ℚ) * initial_average - misread_number = (n : ℚ) * correct_average - correct_number ∧ n = 10 :=
by sorry

end number_of_elements_in_set_l1037_103798


namespace complex_set_equals_zero_two_neg_two_l1037_103727

def complex_set : Set ℂ := {z | ∃ n : ℤ, z = Complex.I ^ n + Complex.I ^ (-n)}

theorem complex_set_equals_zero_two_neg_two : 
  complex_set = {0, 2, -2} :=
sorry

end complex_set_equals_zero_two_neg_two_l1037_103727


namespace profit_percentage_calculation_l1037_103768

/-- 
Given that the cost price is 89% of the selling price, 
prove that the profit percentage is (100/89 - 1) * 100.
-/
theorem profit_percentage_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.89 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/89 - 1) * 100 := by
  sorry

#eval (100/89 - 1) * 100 -- This will output approximately 12.36

end profit_percentage_calculation_l1037_103768


namespace root_inequality_l1037_103737

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) := exp x + x - 2
def g (x : ℝ) := log x + x - 2

-- State the theorem
theorem root_inequality (a b : ℝ) (ha : f a = 0) (hb : g b = 0) :
  f a < f 1 ∧ f 1 < f b := by
  sorry

end

end root_inequality_l1037_103737


namespace find_x_l1037_103701

theorem find_x : ∃ x : ℝ, 121 * x = 75625 ∧ x = 625 := by
  sorry

end find_x_l1037_103701


namespace quadratic_unique_solution_l1037_103799

theorem quadratic_unique_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b^2 + 1/b^2) * x + c = 0) ↔ 
  (c = (1 + Real.sqrt 2) / 2 ∨ c = (1 - Real.sqrt 2) / 2) :=
by sorry

end quadratic_unique_solution_l1037_103799


namespace floor_sum_eval_l1037_103756

theorem floor_sum_eval : ⌊(-7/4 : ℚ)⌋ + ⌊(-3/2 : ℚ)⌋ - ⌊(-5/3 : ℚ)⌋ = -2 := by
  sorry

end floor_sum_eval_l1037_103756


namespace investment_time_calculation_l1037_103775

/-- Investment time calculation for partners P and Q -/
theorem investment_time_calculation 
  (investment_ratio_p : ℝ) 
  (investment_ratio_q : ℝ)
  (profit_ratio_p : ℝ) 
  (profit_ratio_q : ℝ)
  (time_q : ℝ) :
  investment_ratio_p = 7 →
  investment_ratio_q = 5.00001 →
  profit_ratio_p = 7.00001 →
  profit_ratio_q = 10 →
  time_q = 9.999965714374696 →
  ∃ (time_p : ℝ), abs (time_p - 50) < 0.0001 :=
by sorry

end investment_time_calculation_l1037_103775


namespace inequality_proof_l1037_103790

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 :=
by sorry

end inequality_proof_l1037_103790


namespace factor_implies_d_value_l1037_103741

/-- The polynomial Q(x) -/
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

/-- Theorem: If x + 2 is a factor of Q(x), then d = -14 -/
theorem factor_implies_d_value (d : ℝ) :
  (∀ x, Q d x = 0 ↔ x = -2 ∨ (x + 2) * (x^2 - 5*x + 4 - d/2) = 0) →
  d = -14 := by
  sorry

end factor_implies_d_value_l1037_103741


namespace beef_weight_loss_percentage_l1037_103729

theorem beef_weight_loss_percentage 
  (initial_weight : ℝ) 
  (processed_weight : ℝ) 
  (h1 : initial_weight = 1500) 
  (h2 : processed_weight = 750) : 
  (initial_weight - processed_weight) / initial_weight * 100 = 50 := by
sorry

end beef_weight_loss_percentage_l1037_103729


namespace best_of_three_win_probability_l1037_103722

/-- The probability of winning a single game -/
def p : ℚ := 3 / 5

/-- The probability of winning the overall competition in a best-of-three format -/
def win_probability : ℚ :=
  p^2 + 2 * p^2 * (1 - p)

theorem best_of_three_win_probability :
  win_probability = 81 / 125 := by
  sorry

end best_of_three_win_probability_l1037_103722


namespace base_conversion_theorem_l1037_103749

/-- Conversion from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- Given 563 in base 7 equals xy in base 10, prove that (x+y)/9 = 11/9 -/
theorem base_conversion_theorem :
  let n := 563
  let xy := base7ToBase10 n
  let x := xy / 10
  let y := xy % 10
  (x + y) / 9 = 11 / 9 := by sorry

end base_conversion_theorem_l1037_103749


namespace initial_alcohol_percentage_l1037_103742

theorem initial_alcohol_percentage 
  (initial_volume : ℝ) 
  (added_alcohol : ℝ) 
  (added_water : ℝ) 
  (final_percentage : ℝ) :
  initial_volume = 40 →
  added_alcohol = 2.5 →
  added_water = 7.5 →
  final_percentage = 9 →
  ∃ (initial_percentage : ℝ),
    initial_percentage * initial_volume / 100 + added_alcohol = 
    final_percentage * (initial_volume + added_alcohol + added_water) / 100 ∧
    initial_percentage = 5 :=
by sorry

end initial_alcohol_percentage_l1037_103742


namespace max_bouquets_is_37_l1037_103773

/-- Represents the number of flowers available for each type -/
structure FlowerInventory where
  narcissus : ℕ
  chrysanthemum : ℕ
  tulip : ℕ
  lily : ℕ
  rose : ℕ

/-- Represents the constraints for creating a bouquet -/
structure BouquetConstraints where
  min_narcissus : ℕ
  min_chrysanthemum : ℕ
  min_tulip : ℕ
  max_lily_or_rose : ℕ
  max_total : ℕ

/-- Calculates the maximum number of bouquets that can be made -/
def maxBouquets (inventory : FlowerInventory) (constraints : BouquetConstraints) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of bouquets is 37 -/
theorem max_bouquets_is_37 :
  let inventory := FlowerInventory.mk 75 90 50 45 60
  let constraints := BouquetConstraints.mk 2 1 1 3 10
  maxBouquets inventory constraints = 37 := by sorry

end max_bouquets_is_37_l1037_103773


namespace least_common_multiple_2_3_4_5_6_sixty_divisible_by_2_3_4_5_6_least_number_of_marbles_l1037_103734

theorem least_common_multiple_2_3_4_5_6 : ∀ n : ℕ, n > 0 → (2 ∣ n) ∧ (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) → n ≥ 60 := by
  sorry

theorem sixty_divisible_by_2_3_4_5_6 : (2 ∣ 60) ∧ (3 ∣ 60) ∧ (4 ∣ 60) ∧ (5 ∣ 60) ∧ (6 ∣ 60) := by
  sorry

theorem least_number_of_marbles : ∃! n : ℕ, n > 0 ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, m > 0 → (2 ∣ m) ∧ (3 ∣ m) ∧ (4 ∣ m) ∧ (5 ∣ m) ∧ (6 ∣ m) → n ≤ m := by
  sorry

end least_common_multiple_2_3_4_5_6_sixty_divisible_by_2_3_4_5_6_least_number_of_marbles_l1037_103734


namespace linear_function_not_in_first_quadrant_l1037_103725

/-- A linear function that decreases as x increases and satisfies kb > 0 does not pass through the first quadrant -/
theorem linear_function_not_in_first_quadrant
  (k b : ℝ) -- k and b are real numbers
  (h1 : k * b > 0) -- condition: kb > 0
  (h2 : k < 0) -- condition: y decreases as x increases
  : ∀ x y : ℝ, y = k * x + b → ¬(x > 0 ∧ y > 0) := by
  sorry

end linear_function_not_in_first_quadrant_l1037_103725


namespace knight_same_color_probability_l1037_103712

/-- Represents the colors of the chessboard squares -/
inductive ChessColor
| Red
| Green
| Blue

/-- Represents a square on the chessboard -/
structure ChessSquare where
  row : Fin 8
  col : Fin 8
  color : ChessColor

/-- The chessboard with its colored squares -/
def chessboard : Array ChessSquare := sorry

/-- Determines if a knight's move is legal -/
def isLegalKnightMove (start finish : ChessSquare) : Bool := sorry

/-- Calculates the probability of a knight landing on the same color after one move -/
def knightSameColorProbability (board : Array ChessSquare) : ℚ := sorry

/-- The main theorem to prove -/
theorem knight_same_color_probability :
  knightSameColorProbability chessboard = 1/2 := by sorry

end knight_same_color_probability_l1037_103712


namespace sum_after_removal_l1037_103788

def original_series : List ℚ := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]

def removed_terms : List ℚ := [1/8, 1/10]

theorem sum_after_removal :
  (original_series.filter (λ x => x ∉ removed_terms)).sum = 1 := by
  sorry

end sum_after_removal_l1037_103788


namespace cyclic_triples_count_l1037_103785

/-- Represents a round-robin tournament -/
structure Tournament where
  n : ℕ  -- number of teams
  wins : Fin n → ℕ  -- number of wins for each team
  losses : Fin n → ℕ  -- number of losses for each team

/-- The number of sets of three teams with a cyclic winning relationship -/
def cyclic_triples (t : Tournament) : ℕ := sorry

/-- Main theorem about the number of cyclic triples in the specific tournament -/
theorem cyclic_triples_count (t : Tournament) 
  (h1 : t.n > 0)
  (h2 : ∀ i : Fin t.n, t.wins i = 9)
  (h3 : ∀ i : Fin t.n, t.losses i = 9)
  (h4 : ∀ i j : Fin t.n, i ≠ j → (t.wins i + t.losses i = t.wins j + t.losses j)) :
  cyclic_triples t = 969 := by sorry

end cyclic_triples_count_l1037_103785


namespace banker_cannot_guarantee_2kg_l1037_103721

/-- Represents the state of the banker's sand and exchange rates -/
structure SandState where
  g : ℕ -- Exchange rate for gold
  p : ℕ -- Exchange rate for platinum
  G : ℚ -- Amount of gold sand in kg
  P : ℚ -- Amount of platinum sand in kg

/-- Calculates the invariant metric S for a given SandState -/
def calcMetric (state : SandState) : ℚ :=
  state.G * state.p + state.P * state.g

/-- Represents the daily change in exchange rates -/
inductive DailyChange
  | decreaseG
  | decreaseP

/-- Applies a daily change to the SandState -/
def applyDailyChange (state : SandState) (change : DailyChange) : SandState :=
  match change with
  | DailyChange.decreaseG => { state with g := state.g - 1 }
  | DailyChange.decreaseP => { state with p := state.p - 1 }

/-- Theorem stating that the banker cannot guarantee 2 kg of each sand type after 2000 days -/
theorem banker_cannot_guarantee_2kg (initialState : SandState)
  (h_initial_g : initialState.g = 1001)
  (h_initial_p : initialState.p = 1001)
  (h_initial_G : initialState.G = 1)
  (h_initial_P : initialState.P = 1) :
  ¬ ∃ (finalState : SandState),
    (∃ (changes : List DailyChange),
      changes.length = 2000 ∧
      finalState = changes.foldl applyDailyChange initialState ∧
      finalState.g = 1 ∧ finalState.p = 1) ∧
    finalState.G ≥ 2 ∧ finalState.P ≥ 2 :=
  sorry

#check banker_cannot_guarantee_2kg

end banker_cannot_guarantee_2kg_l1037_103721


namespace simplify_expression_l1037_103719

theorem simplify_expression (x : ℝ) : (2 * x + 30) + (150 * x + 45) + 5 = 152 * x + 80 := by
  sorry

end simplify_expression_l1037_103719


namespace product_real_implies_ratio_l1037_103791

def complex (a b : ℝ) : ℂ := a + b * Complex.I

theorem product_real_implies_ratio (a b : ℝ) (hb : b ≠ 0) :
  let z₁ : ℂ := 2 + 3 * Complex.I
  let z₂ : ℂ := complex a b
  (z₁ * z₂).im = 0 → a / b = -2 / 3 := by
  sorry

end product_real_implies_ratio_l1037_103791


namespace parallel_vectors_ratio_l1037_103745

theorem parallel_vectors_ratio (θ : ℝ) : 
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (1, 3)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2 := by
sorry

end parallel_vectors_ratio_l1037_103745


namespace hyperbola_equation_correct_l1037_103778

/-- Represents a hyperbola with equation ax² - by² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → a * x^2 - b * y^2 = 1

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def has_focus (h : Hyperbola) (p : Point) : Prop :=
  ∃ c : ℝ, c^2 = 1 / h.a + 1 / h.b ∧ p.y^2 = c^2

def has_same_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a / h1.b = h2.a / h2.b

theorem hyperbola_equation_correct (h1 h2 : Hyperbola) (p : Point) :
  h1.a = 1/24 ∧ h1.b = 1/12 ∧
  h2.a = 1/2 ∧ h2.b = 1 ∧
  p.x = 0 ∧ p.y = 6 ∧
  has_focus h1 p ∧
  has_same_asymptotes h1 h2 :=
by sorry

end hyperbola_equation_correct_l1037_103778


namespace equation_solutions_l1037_103717

theorem equation_solutions :
  (∃ x : ℚ, (5*x - 1)/4 = (3*x + 1)/2 - (2 - x)/3 ↔ x = -1/7) ∧
  (∃ x : ℚ, (3*x + 2)/2 - 1 = (2*x - 1)/4 - (2*x + 1)/5 ↔ x = -9/28) := by
  sorry

end equation_solutions_l1037_103717


namespace zoey_finishes_on_friday_l1037_103740

def days_to_read (n : ℕ) : ℕ := n + 1

def total_days (sets : ℕ) : ℕ :=
  (List.range sets).map (λ i => days_to_read (i + 1)) |>.sum

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem zoey_finishes_on_friday :
  let sets := 8
  let start_day := 3  -- Wednesday (0 = Sunday, 1 = Monday, ..., 6 = Saturday)
  day_of_week start_day (total_days sets) = 5  -- Friday
  := by sorry

end zoey_finishes_on_friday_l1037_103740


namespace equation_solution_l1037_103748

theorem equation_solution : ∃ x : ℝ, (5*x + 9*x = 350 - 10*(x - 4)) ∧ x = 16.25 := by
  sorry

end equation_solution_l1037_103748


namespace unique_n_for_prime_roots_l1037_103763

/-- Determines if a natural number is prime -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

/-- The quadratic equation as a function of x and n -/
def quadraticEq (x n : ℕ) : ℤ :=
  2 * x^2 - 8*n*x + 10*x - n^2 + 35*n - 76

theorem unique_n_for_prime_roots :
  ∃! n : ℕ, ∃ x₁ x₂ : ℕ,
    x₁ ≠ x₂ ∧
    isPrime x₁ ∧
    isPrime x₂ ∧
    quadraticEq x₁ n = 0 ∧
    quadraticEq x₂ n = 0 ∧
    n = 3 ∧
    x₁ = 2 ∧
    x₂ = 5 :=
sorry

end unique_n_for_prime_roots_l1037_103763


namespace x_fourth_minus_inverse_x_fourth_l1037_103754

theorem x_fourth_minus_inverse_x_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end x_fourth_minus_inverse_x_fourth_l1037_103754


namespace race_theorem_l1037_103724

def race_problem (john_speed : ℝ) (race_distance : ℝ) (winning_margin : ℝ) : Prop :=
  let john_time := race_distance / john_speed * 60
  let next_fastest_time := john_time + winning_margin
  next_fastest_time = 23

theorem race_theorem :
  race_problem 15 5 3 := by
  sorry

end race_theorem_l1037_103724


namespace polar_to_rectangular_conversion_l1037_103794

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 3 * Real.sqrt 2 / 2 ∧ y = 3 * Real.sqrt 2 / 2 := by
  sorry

end polar_to_rectangular_conversion_l1037_103794


namespace comparison_and_inequality_l1037_103732

theorem comparison_and_inequality (x y m : ℝ) 
  (h1 : x > y) (h2 : y > 0) (h3 : m > 0) : 
  y / x < (y + m) / (x + m) ∧ Real.sqrt (x * y) * (2 - Real.sqrt (x * y)) ≤ 1 := by
  sorry

end comparison_and_inequality_l1037_103732


namespace five_digit_palindromes_count_l1037_103786

/-- A five-digit palindromic number -/
def FiveDigitPalindrome (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * b + a

/-- The count of five-digit palindromic numbers -/
def CountFiveDigitPalindromes : ℕ := 90

theorem five_digit_palindromes_count :
  (∀ n : ℕ, 10000 ≤ n ∧ n < 100000 →
    (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ n = FiveDigitPalindrome a b c) ↔
    (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ n = 10000 * a + 1000 * b + 100 * c + 10 * b + a)) →
  CountFiveDigitPalindromes = (9 : ℕ) * (10 : ℕ) * (1 : ℕ) :=
sorry

end five_digit_palindromes_count_l1037_103786


namespace stamps_bought_theorem_l1037_103781

/-- The total number of stamps bought by Evariste and Sophie -/
def total_stamps (x y : ℕ) : ℕ := x + y

/-- The cost of Evariste's stamps in pence -/
def evariste_cost : ℕ := 110

/-- The cost of Sophie's stamps in pence -/
def sophie_cost : ℕ := 70

/-- The total amount spent in pence -/
def total_spent : ℕ := 1000

theorem stamps_bought_theorem (x y : ℕ) :
  x * evariste_cost + y * sophie_cost = total_spent →
  total_stamps x y = 12 := by
  sorry

#check stamps_bought_theorem

end stamps_bought_theorem_l1037_103781


namespace garrison_reinforcement_size_l1037_103766

/-- Calculates the size of reinforcement given garrison provisions information -/
theorem garrison_reinforcement_size
  (initial_size : ℕ)
  (initial_duration : ℕ)
  (initial_consumption : ℚ)
  (time_before_reinforcement : ℕ)
  (new_consumption : ℚ)
  (additional_duration : ℕ)
  (h1 : initial_size = 2000)
  (h2 : initial_duration = 40)
  (h3 : initial_consumption = 3/2)
  (h4 : time_before_reinforcement = 20)
  (h5 : new_consumption = 2)
  (h6 : additional_duration = 10) :
  ∃ (reinforcement_size : ℕ),
    reinforcement_size = 1500 ∧
    (initial_size * initial_consumption * initial_duration : ℚ) =
    (initial_size * initial_consumption * time_before_reinforcement +
     (initial_size * initial_consumption + reinforcement_size * new_consumption) * additional_duration : ℚ) :=
by sorry

end garrison_reinforcement_size_l1037_103766


namespace floor_equation_solution_l1037_103780

theorem floor_equation_solution (x : ℝ) :
  ⌊⌊3 * x⌋ + (1 : ℝ) / 2⌋ = ⌊x + 3⌋ ↔ 4 / 3 ≤ x ∧ x < 2 :=
sorry

end floor_equation_solution_l1037_103780


namespace min_value_of_reciprocal_sum_l1037_103709

theorem min_value_of_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (2 * a * (-1) - b * 2 + 2 = 0) → -- Line passes through circle center (-1, 2)
  (∀ x y : ℝ, 2 * a * x - b * y + 2 = 0 → x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (2 * a' * (-1) - b' * 2 + 2 = 0) → 
    (1/a + 1/b) ≤ (1/a' + 1/b')) →
  1/a + 1/b = 4 :=
by sorry

end min_value_of_reciprocal_sum_l1037_103709


namespace find_a_l1037_103715

def A : Set ℝ := {x | x^2 + 6*x < 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a-2)*x - 2*a < 0}

theorem find_a : 
  A ∪ B a = {x : ℝ | -6 < x ∧ x < 5} → a = 5 := by
sorry

end find_a_l1037_103715


namespace complex_equation_sum_of_squares_l1037_103758

theorem complex_equation_sum_of_squares (a b : ℝ) (i : ℂ) : 
  i * i = -1 → 
  (a + i) / i = b + i → 
  a^2 + b^2 = 2 := by
  sorry

end complex_equation_sum_of_squares_l1037_103758


namespace factoring_transformation_l1037_103728

theorem factoring_transformation (y : ℝ) : 4 * y^2 - 4 * y + 1 = (2 * y - 1)^2 := by
  sorry

end factoring_transformation_l1037_103728


namespace min_rowers_theorem_l1037_103723

/-- Represents a lyamzik with a weight --/
structure Lyamzik where
  weight : Nat

/-- Represents the boat used for crossing --/
structure Boat where
  maxWeight : Nat

/-- Represents the river crossing scenario --/
structure RiverCrossing where
  lyamziks : List Lyamzik
  boat : Boat
  maxRowsPerLyamzik : Nat

/-- The minimum number of lyamziks required to row --/
def minRowersRequired (rc : RiverCrossing) : Nat :=
  12

theorem min_rowers_theorem (rc : RiverCrossing) 
  (h1 : rc.lyamziks.length = 28)
  (h2 : (rc.lyamziks.filter (fun l => l.weight = 2)).length = 7)
  (h3 : (rc.lyamziks.filter (fun l => l.weight = 3)).length = 7)
  (h4 : (rc.lyamziks.filter (fun l => l.weight = 4)).length = 7)
  (h5 : (rc.lyamziks.filter (fun l => l.weight = 5)).length = 7)
  (h6 : rc.boat.maxWeight = 10)
  (h7 : rc.maxRowsPerLyamzik = 2) :
  minRowersRequired rc ≥ 12 := by
  sorry

#check min_rowers_theorem

end min_rowers_theorem_l1037_103723


namespace cube_root_product_l1037_103755

theorem cube_root_product : (4^9 * 5^6 * 7^3 : ℝ)^(1/3) = 11200 := by sorry

end cube_root_product_l1037_103755


namespace smallest_multiple_of_84_with_6_and_7_l1037_103730

def is_multiple_of_84 (n : ℕ) : Prop := n % 84 = 0

def contains_only_6_and_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 6 ∨ d = 7

theorem smallest_multiple_of_84_with_6_and_7 :
  (is_multiple_of_84 76776) ∧
  (contains_only_6_and_7 76776) ∧
  (∀ n : ℕ, n < 76776 → ¬(is_multiple_of_84 n ∧ contains_only_6_and_7 n)) :=
sorry

end smallest_multiple_of_84_with_6_and_7_l1037_103730


namespace sum_of_digits_3n_l1037_103761

/-- Sum of decimal digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Given a natural number n where the sum of its digits is 100 and
    the sum of digits of 44n is 800, prove that the sum of digits of 3n is 300 -/
theorem sum_of_digits_3n (n : ℕ) 
  (h1 : sumOfDigits n = 100) 
  (h2 : sumOfDigits (44 * n) = 800) : 
  sumOfDigits (3 * n) = 300 := by sorry

end sum_of_digits_3n_l1037_103761


namespace similar_triangles_height_l1037_103792

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 3 →
  area_ratio = 4 →
  ∃ h_large : ℝ, h_large = 6 ∧ h_large / h_small = Real.sqrt area_ratio :=
by sorry

end similar_triangles_height_l1037_103792


namespace prob_different_colors_specific_l1037_103789

/-- The probability of drawing two chips of different colors from a bag --/
def prob_different_colors (blue red yellow : ℕ) : ℚ :=
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / (total - 1)
  let prob_not_red := (blue + yellow) / (total - 1)
  let prob_not_yellow := (blue + red) / (total - 1)
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow

/-- Theorem: The probability of drawing two chips of different colors from a bag with 7 blue, 6 red, and 5 yellow chips is 122/153 --/
theorem prob_different_colors_specific : prob_different_colors 7 6 5 = 122 / 153 := by
  sorry

end prob_different_colors_specific_l1037_103789


namespace geometric_sum_abs_l1037_103702

def geometric_sequence (n : ℕ) (a₁ : ℝ) (r : ℝ) : ℝ := a₁ * r^(n-1)

theorem geometric_sum_abs (a₁ r : ℝ) (h : a₁ = 1 ∧ r = -2) :
  let a := geometric_sequence
  a 1 a₁ r + |a 2 a₁ r| + |a 3 a₁ r| + a 4 a₁ r = 15 := by sorry

end geometric_sum_abs_l1037_103702


namespace remaining_garden_area_is_48_l1037_103782

/-- The area of a rectangle with given length and width -/
def rectangleArea (length width : ℝ) : ℝ := length * width

/-- The dimensions of the large garden -/
def largeGardenLength : ℝ := 10
def largeGardenWidth : ℝ := 6

/-- The dimensions of the small plot -/
def smallPlotLength : ℝ := 4
def smallPlotWidth : ℝ := 3

/-- The remaining garden area after removing the small plot -/
def remainingGardenArea : ℝ :=
  rectangleArea largeGardenLength largeGardenWidth -
  rectangleArea smallPlotLength smallPlotWidth

theorem remaining_garden_area_is_48 :
  remainingGardenArea = 48 := by sorry

end remaining_garden_area_is_48_l1037_103782
