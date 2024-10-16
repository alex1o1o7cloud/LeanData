import Mathlib

namespace NUMINAMATH_CALUDE_find_A_in_three_digit_sum_l1997_199701

theorem find_A_in_three_digit_sum (A B : ℕ) : 
  (100 ≤ A * 100 + 70 + B) ∧ 
  (A * 100 + 70 + B < 1000) ∧ 
  (32 + A * 100 + 70 + B = 705) → 
  A = 6 := by
sorry

end NUMINAMATH_CALUDE_find_A_in_three_digit_sum_l1997_199701


namespace NUMINAMATH_CALUDE_octal_135_to_binary_l1997_199740

/-- Converts an octal digit to its binary representation --/
def octal_to_binary_digit (d : Nat) : Nat :=
  match d with
  | 0 => 0
  | 1 => 1
  | 2 => 10
  | 3 => 11
  | 4 => 100
  | 5 => 101
  | 6 => 110
  | 7 => 111
  | _ => 0  -- Default case, should not occur for valid octal digits

/-- Converts an octal number to its binary representation --/
def octal_to_binary (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  octal_to_binary_digit hundreds * 1000000 +
  octal_to_binary_digit tens * 1000 +
  octal_to_binary_digit ones

theorem octal_135_to_binary : octal_to_binary 135 = 1011101 := by
  sorry

end NUMINAMATH_CALUDE_octal_135_to_binary_l1997_199740


namespace NUMINAMATH_CALUDE_altitude_feet_locus_l1997_199760

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The locus of altitude feet for a varying right triangle -/
def altitudeFeetLocus (S₁ S₂ : Circle) (A : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- The arc or segment of the circle with diameter AM -/
def circleArcOrSegment (A M : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- External center of similarity of two circles -/
def externalCenterOfSimilarity (S₁ S₂ : Circle) : ℝ × ℝ :=
  sorry

/-- Main theorem: The locus of altitude feet is an arc or segment of circle with diameter AM -/
theorem altitude_feet_locus (S₁ S₂ : Circle) (A : ℝ × ℝ) :
  altitudeFeetLocus S₁ S₂ A = 
  circleArcOrSegment A (externalCenterOfSimilarity S₁ S₂) :=
by sorry

end NUMINAMATH_CALUDE_altitude_feet_locus_l1997_199760


namespace NUMINAMATH_CALUDE_total_seeds_planted_l1997_199771

/-- The number of seeds planted in each flower bed -/
def seeds_per_bed : ℕ := 10

/-- The number of flower beds -/
def number_of_beds : ℕ := 6

/-- The total number of seeds planted -/
def total_seeds : ℕ := seeds_per_bed * number_of_beds

theorem total_seeds_planted : total_seeds = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_planted_l1997_199771


namespace NUMINAMATH_CALUDE_rational_square_root_condition_l1997_199793

theorem rational_square_root_condition (n : ℤ) :
  n ≥ 3 →
  (∃ (q : ℚ), q^2 = (n^2 - 5) / (n + 1)) ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_rational_square_root_condition_l1997_199793


namespace NUMINAMATH_CALUDE_intersection_sine_value_l1997_199796

theorem intersection_sine_value (x₀ : Real) (y₀ : Real) :
  x₀ ∈ Set.Ioo 0 (π / 2) →
  y₀ = 3 * Real.cos x₀ →
  y₀ = 8 * Real.tan x₀ →
  Real.sin x₀ = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_sine_value_l1997_199796


namespace NUMINAMATH_CALUDE_complex_product_QED_l1997_199755

theorem complex_product_QED (Q E D : ℂ) : 
  Q = 7 + 3 * Complex.I → 
  E = 2 * Complex.I → 
  D = 7 - 3 * Complex.I → 
  Q * E * D = 116 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_QED_l1997_199755


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1997_199797

/-- A geometric sequence with a_3 = 4 and a_6 = 1/2 has a common ratio of 1/2. -/
theorem geometric_sequence_common_ratio : ∀ (a : ℕ → ℝ), 
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 3 = 4 →                                  -- Condition 1
  a 6 = 1 / 2 →                              -- Condition 2
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ q = 1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1997_199797


namespace NUMINAMATH_CALUDE_relationship_abc_l1997_199799

noncomputable def a : ℝ := (1.1 : ℝ) ^ (0.1 : ℝ)
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := Real.log (Real.sqrt 3 / 3) / Real.log (1/3)

theorem relationship_abc : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1997_199799


namespace NUMINAMATH_CALUDE_strawberry_picking_l1997_199773

theorem strawberry_picking (basket_capacity : ℕ) (picked_ratio : ℚ) : 
  basket_capacity = 60 → 
  picked_ratio = 4/5 → 
  (basket_capacity / picked_ratio : ℚ) * 5 = 75 := by
sorry

end NUMINAMATH_CALUDE_strawberry_picking_l1997_199773


namespace NUMINAMATH_CALUDE_equation_condition_l1997_199729

theorem equation_condition (a b c : ℤ) :
  a * (a - b) + b * (b - c) + c * (c - a) = 0 ↔ 
  (a = c ∧ b - 2 = c ∧ a = 0 ∧ b = 0 ∧ c = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_condition_l1997_199729


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1997_199732

theorem sin_alpha_value (α : Real) 
  (h : Real.cos (α - π / 2) = 1 / 3) : 
  Real.sin α = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1997_199732


namespace NUMINAMATH_CALUDE_square_root_of_2m_minus_n_is_2_l1997_199706

theorem square_root_of_2m_minus_n_is_2 
  (m n : ℝ) 
  (eq1 : m * 2 + n * 1 = 8) 
  (eq2 : n * 2 - m * 1 = 1) : 
  Real.sqrt (2 * m - n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_2m_minus_n_is_2_l1997_199706


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1997_199747

theorem sqrt_equation_solution (a x : ℝ) (ha : 0 < a ∧ a ≤ 1) (hx : x ≥ 1) :
  Real.sqrt (x + Real.sqrt x) - Real.sqrt (x - Real.sqrt x) = (a + 1) * Real.sqrt (x / (x + Real.sqrt x)) →
  x = ((a^2 + 1) / (2*a))^2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1997_199747


namespace NUMINAMATH_CALUDE_correct_ranking_is_cab_l1997_199764

-- Define the teams
inductive Team
| A
| B
| C

-- Define the rankings
def Ranking := Fin 3 → Team

-- Define the prediction type
structure Prediction where
  team1 : Team
  place1 : Fin 3
  team2 : Team
  place2 : Fin 3

-- Define the predictions
def liMing : Prediction := { team1 := Team.A, place1 := 0, team2 := Team.B, place2 := 2 }
def zhangHua : Prediction := { team1 := Team.A, place1 := 2, team2 := Team.C, place2 := 0 }
def wangQiang : Prediction := { team1 := Team.C, place1 := 1, team2 := Team.B, place2 := 2 }

-- Define a function to check if a prediction is half correct
def isHalfCorrect (p : Prediction) (r : Ranking) : Prop :=
  (r p.place1 = p.team1) ≠ (r p.place2 = p.team2)

-- Main theorem
theorem correct_ranking_is_cab :
  ∃! r : Ranking,
    isHalfCorrect liMing r ∧
    isHalfCorrect zhangHua r ∧
    isHalfCorrect wangQiang r ∧
    r 0 = Team.C ∧
    r 1 = Team.A ∧
    r 2 = Team.B :=
sorry

end NUMINAMATH_CALUDE_correct_ranking_is_cab_l1997_199764


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1997_199754

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + (3*x - 5) + 2*x + 18 + (2*x + 6)) / 5 = 30 → x = 15.125 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1997_199754


namespace NUMINAMATH_CALUDE_trivia_team_points_l1997_199761

/-- Given a trivia team with the following properties:
  * total_members: The total number of team members
  * absent_members: The number of members who didn't show up
  * points_per_member: The number of points scored by each member who showed up
  * total_points: The total points scored by the team

  This theorem proves that the total points scored by the team is equal to
  the product of the number of members who showed up and the points per member.
-/
theorem trivia_team_points 
  (total_members : ℕ) 
  (absent_members : ℕ) 
  (points_per_member : ℕ) 
  (total_points : ℕ) 
  (h1 : total_members = 14) 
  (h2 : absent_members = 7) :
  total_points = (total_members - absent_members) * points_per_member := by
sorry

end NUMINAMATH_CALUDE_trivia_team_points_l1997_199761


namespace NUMINAMATH_CALUDE_root_product_equals_eight_l1997_199725

theorem root_product_equals_eight : 
  (64 : ℝ) ^ (1/6) * (8 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_eight_l1997_199725


namespace NUMINAMATH_CALUDE_goldbach_138_largest_diff_l1997_199708

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_138_largest_diff :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 138 ∧ 
    p ≠ q ∧
    ∀ (r s : ℕ), is_prime r → is_prime s → r + s = 138 → r ≠ s → 
      (max r s - min r s) ≤ (max p q - min p q) ∧
    (max p q - min p q) = 124 :=
sorry

end NUMINAMATH_CALUDE_goldbach_138_largest_diff_l1997_199708


namespace NUMINAMATH_CALUDE_max_distance_complex_l1997_199792

theorem max_distance_complex (w : ℂ) (h : Complex.abs w = 3) :
  ∃ (max_dist : ℝ), max_dist = 9 * Real.sqrt 61 + 81 ∧
    ∀ (z : ℂ), Complex.abs z = 3 → Complex.abs ((6 + 5*Complex.I)*z^2 - z^4) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l1997_199792


namespace NUMINAMATH_CALUDE_triangle_side_length_proof_l1997_199749

noncomputable def triangle_side_length (A B C : Real) : Real :=
  let AB : Real := 1
  let AC : Real := 2
  let cos_B_plus_sin_C : Real := 1
  (2 * Real.sqrt 21 + 3) / 5

theorem triangle_side_length_proof (A B C : Real) :
  let AB : Real := 1
  let AC : Real := 2
  let cos_B_plus_sin_C : Real := 1
  triangle_side_length A B C = (2 * Real.sqrt 21 + 3) / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_proof_l1997_199749


namespace NUMINAMATH_CALUDE_bike_shop_theorem_l1997_199770

/-- Represents the sales and pricing information for bike types A and B -/
structure BikeShop where
  lastYearRevenueA : ℕ
  priceDrop : ℕ
  revenueDecrease : ℚ
  totalNewBikes : ℕ
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceB : ℕ

/-- Calculates the selling price of type A bikes this year -/
def sellingPriceA (shop : BikeShop) : ℕ := sorry

/-- Calculates the maximum profit and optimal purchase quantities -/
def maxProfit (shop : BikeShop) : ℕ × ℕ × ℕ := sorry

/-- Main theorem stating the correct selling price and maximum profit -/
theorem bike_shop_theorem (shop : BikeShop) 
  (h1 : shop.lastYearRevenueA = 50000)
  (h2 : shop.priceDrop = 400)
  (h3 : shop.revenueDecrease = 1/5)
  (h4 : shop.totalNewBikes = 60)
  (h5 : shop.purchasePriceA = 1100)
  (h6 : shop.purchasePriceB = 1400)
  (h7 : shop.sellingPriceB = 2000) :
  sellingPriceA shop = 1600 ∧ 
  maxProfit shop = (34000, 20, 40) := by sorry


end NUMINAMATH_CALUDE_bike_shop_theorem_l1997_199770


namespace NUMINAMATH_CALUDE_square_area_calculation_l1997_199772

theorem square_area_calculation (s r l : ℝ) : 
  l = (2/5) * r →
  r = s →
  l * 10 = 200 →
  s^2 = 2500 := by sorry

end NUMINAMATH_CALUDE_square_area_calculation_l1997_199772


namespace NUMINAMATH_CALUDE_max_s_value_l1997_199763

theorem max_s_value (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_products_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) :
  s ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_s_value_l1997_199763


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1997_199795

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 → Nat.gcd a b = 4 → Nat.lcm a b = 80 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1997_199795


namespace NUMINAMATH_CALUDE_comic_cost_l1997_199787

theorem comic_cost (initial_money : ℕ) (comics_bought : ℕ) (money_left : ℕ) : 
  initial_money = 87 → comics_bought = 8 → money_left = 55 → 
  (initial_money - money_left) / comics_bought = 4 := by
sorry

end NUMINAMATH_CALUDE_comic_cost_l1997_199787


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1997_199783

-- Define set A
def A : Set ℝ := {x | x^2 + x - 2 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x > 0, y = Real.log x / Real.log 2}

-- Define the complement of A in ℝ
def complement_A : Set ℝ := {x | x ∉ A}

-- Theorem statement
theorem complement_A_intersect_B : complement_A ∩ B = Set.Icc (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1997_199783


namespace NUMINAMATH_CALUDE_min_sum_squares_l1997_199728

theorem min_sum_squares (a b c d e f g h : ℤ) : 
  a ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  b ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  c ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  d ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  e ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  f ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  g ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  h ∈ ({-7, -5, -3, -1, 0, 2, 4, 5} : Set ℤ) →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g → a ≠ h →
  b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g → b ≠ h →
  c ≠ d → c ≠ e → c ≠ f → c ≠ g → c ≠ h →
  d ≠ e → d ≠ f → d ≠ g → d ≠ h →
  e ≠ f → e ≠ g → e ≠ h →
  f ≠ g → f ≠ h →
  g ≠ h →
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 34 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1997_199728


namespace NUMINAMATH_CALUDE_range_of_3x_plus_2y_l1997_199789

theorem range_of_3x_plus_2y (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : -1 ≤ y ∧ y ≤ 4) :
  ∃ (z : ℝ), z = 3*x + 2*y ∧ 1 ≤ z ∧ z ≤ 17 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), (1 ≤ a ∧ a ≤ 3) ∧ (-1 ≤ b ∧ b ≤ 4) ∧ w = 3*a + 2*b) → 1 ≤ w ∧ w ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_range_of_3x_plus_2y_l1997_199789


namespace NUMINAMATH_CALUDE_system_and_expression_solution_l1997_199735

theorem system_and_expression_solution :
  -- System of equations
  (∃ (x y : ℝ), 2*x + y = 4 ∧ x + 2*y = 5 ∧ x = 1 ∧ y = 2) ∧
  -- Simplified expression evaluation
  (let x : ℝ := -2; (x^2 + 1) / x = -5/2) :=
sorry

end NUMINAMATH_CALUDE_system_and_expression_solution_l1997_199735


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l1997_199713

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetry_coordinates :
  let A : Point := ⟨1, -2⟩
  let A' : Point := ⟨-1, 2⟩
  symmetricToOrigin A A' :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l1997_199713


namespace NUMINAMATH_CALUDE_five_digit_square_number_l1997_199739

theorem five_digit_square_number : ∃! n : ℕ, 
  (n * n ≥ 10000) ∧ 
  (n * n < 100000) ∧ 
  (n * n / 10000 = 2) ∧ 
  ((n * n / 10) % 10 = 5) ∧ 
  (∃ m : ℕ, n * n = m * m) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_square_number_l1997_199739


namespace NUMINAMATH_CALUDE_probability_6_or_7_heads_in_8_flips_l1997_199775

def n : ℕ := 8  -- number of coin flips

-- Define the probability of getting exactly k heads in n flips
def prob_k_heads (k : ℕ) : ℚ :=
  (n.choose k) / 2^n

-- Define the probability of getting exactly 6 or 7 heads in n flips
def prob_6_or_7_heads : ℚ :=
  prob_k_heads 6 + prob_k_heads 7

-- Theorem statement
theorem probability_6_or_7_heads_in_8_flips :
  prob_6_or_7_heads = 9 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_6_or_7_heads_in_8_flips_l1997_199775


namespace NUMINAMATH_CALUDE_point_not_in_plane_l1997_199752

def in_plane (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_plane :
  ¬(in_plane 2 0) ∧ 
  in_plane 0 0 ∧ 
  in_plane 1 1 ∧ 
  in_plane 0 2 :=
sorry

end NUMINAMATH_CALUDE_point_not_in_plane_l1997_199752


namespace NUMINAMATH_CALUDE_train_crossing_time_l1997_199794

theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 330 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 44 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1997_199794


namespace NUMINAMATH_CALUDE_jamies_mean_score_l1997_199778

def scores : List ℕ := [80, 85, 90, 95, 100, 105]

theorem jamies_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ alex_scores jamie_scores : List ℕ, 
        alex_scores.length = 3 ∧ 
        jamie_scores.length = 3 ∧ 
        scores = alex_scores ++ jamie_scores)
  (h3 : ∃ alex_scores : List ℕ, 
        alex_scores.length = 3 ∧ 
        alex_scores.sum / alex_scores.length = 85)
  : ∃ jamie_scores : List ℕ,
    jamie_scores.length = 3 ∧
    jamie_scores.sum / jamie_scores.length = 100 := by
  sorry

end NUMINAMATH_CALUDE_jamies_mean_score_l1997_199778


namespace NUMINAMATH_CALUDE_sugar_price_increase_l1997_199731

theorem sugar_price_increase (initial_price : ℝ) (consumption_reduction : ℝ) (new_price : ℝ) : 
  initial_price = 6 →
  consumption_reduction = 19.999999999999996 →
  (1 - consumption_reduction / 100) * new_price = initial_price →
  new_price = 7.5 := by
sorry

end NUMINAMATH_CALUDE_sugar_price_increase_l1997_199731


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1997_199777

/-- Given a curve y = x^3 + ax + b and a line y = 2x + 1 tangent to the curve at the point (1, 3),
    prove that the value of b is 3. -/
theorem tangent_line_problem (a b : ℝ) : 
  (∀ x : ℝ, x^3 + a*x + b = 2*x + 1 → x = 1) →
  1^3 + a*1 + b = 3 →
  (3*(1^2) + a = 2) →
  b = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l1997_199777


namespace NUMINAMATH_CALUDE_min_sum_quadratic_roots_l1997_199786

theorem min_sum_quadratic_roots (a b : ℕ+) (h1 : ∃ x y : ℝ, 
  x ≠ y ∧ -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ 
  a * x^2 + b * x + 1 = 0 ∧ a * y^2 + b * y + 1 = 0) : 
  (∀ a' b' : ℕ+, (∃ x y : ℝ, 
    x ≠ y ∧ -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ 
    a' * x^2 + b' * x + 1 = 0 ∧ a' * y^2 + b' * y + 1 = 0) → 
  (a'.val + b'.val : ℕ) ≥ (a.val + b.val)) ∧ 
  (a.val + b.val : ℕ) = 10 := by sorry

end NUMINAMATH_CALUDE_min_sum_quadratic_roots_l1997_199786


namespace NUMINAMATH_CALUDE_prob_white_then_black_l1997_199722

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents the bag of balls -/
def Bag := Finset Color

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of black balls in the bag -/
def black_balls : ℕ := 2

/-- The bag containing the balls -/
def initial_bag : Bag := sorry

/-- The probability of drawing a white ball first and a black ball second without replacement -/
theorem prob_white_then_black : 
  (white_balls / total_balls) * (black_balls / (total_balls - 1)) = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_prob_white_then_black_l1997_199722


namespace NUMINAMATH_CALUDE_counterexample_exists_l1997_199766

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ Nat.Prime (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1997_199766


namespace NUMINAMATH_CALUDE_pizza_toppings_l1997_199756

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) :
  total_slices = 12 →
  pepperoni_slices = 6 →
  mushroom_slices = 10 →
  pepperoni_slices + mushroom_slices ≥ total_slices →
  ∃ (both_toppings : ℕ),
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 4 :=
by
  sorry

#check pizza_toppings

end NUMINAMATH_CALUDE_pizza_toppings_l1997_199756


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1997_199757

theorem sum_of_fractions : (6 : ℚ) / 5 + (1 : ℚ) / 10 = (13 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1997_199757


namespace NUMINAMATH_CALUDE_submerged_sphere_segment_height_l1997_199702

/-- 
Theorem: For a homogeneous spherical segment of radius r floating in water, 
the height of the submerged portion m is equal to r/2 * (3 - √5) when it 
submerges up to the edge of its base spherical cap.
-/
theorem submerged_sphere_segment_height 
  (r : ℝ) -- radius of the sphere
  (h_pos : r > 0) -- assumption that radius is positive
  : ∃ m : ℝ, 
    -- m is the height of the submerged portion
    -- Volume of spherical sector
    (2 * π * m^3 / 3 = 
    -- Volume of submerged spherical segment
    π * m^2 * (3*r - m) / 3) ∧ 
    -- m is less than r (physical constraint)
    m < r ∧ 
    -- m equals the derived formula
    m = r/2 * (3 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_submerged_sphere_segment_height_l1997_199702


namespace NUMINAMATH_CALUDE_five_Y_three_Z_one_eq_one_l1997_199790

/-- Custom operator Y Z -/
def Y_Z (a b c : ℝ) : ℝ := (a - b - c)^2

/-- Theorem stating that 5 Y 3 Z 1 = 1 -/
theorem five_Y_three_Z_one_eq_one : Y_Z 5 3 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_five_Y_three_Z_one_eq_one_l1997_199790


namespace NUMINAMATH_CALUDE_impossible_shot_l1997_199720

-- Define the elliptical billiard table
structure EllipticalTable where
  foci : Pointℝ × Pointℝ

-- Define the balls
structure Ball where
  position : Pointℝ

-- Define the properties of the problem
def is_on_edge (table : EllipticalTable) (ball : Ball) : Prop := sorry

def is_on_focal_segment (table : EllipticalTable) (ball : Ball) : Prop := sorry

def bounces_and_hits (table : EllipticalTable) (ball_A ball_B : Ball) : Prop := sorry

def crosses_focal_segment_before_bounce (table : EllipticalTable) (ball : Ball) : Prop := sorry

-- State the theorem
theorem impossible_shot (table : EllipticalTable) (ball_A ball_B : Ball) :
  is_on_edge table ball_A ∧
  is_on_focal_segment table ball_B ∧
  bounces_and_hits table ball_A ball_B ∧
  ¬crosses_focal_segment_before_bounce table ball_A →
  False := by sorry

end NUMINAMATH_CALUDE_impossible_shot_l1997_199720


namespace NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l1997_199719

/-- The decimal representation of 4/17 has a repeating block of 235294117647 -/
theorem repetend_of_four_seventeenths : 
  ∃ (n : ℕ), (4 : ℚ) / 17 = (n : ℚ) / 999999999999 ∧ n = 235294117647 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l1997_199719


namespace NUMINAMATH_CALUDE_nell_ace_cards_l1997_199744

/-- The number of baseball cards Nell has after giving some to Jeff -/
def remaining_baseball_cards : ℕ := 111

/-- The difference between Ace cards and baseball cards Nell has now -/
def ace_baseball_difference : ℕ := 265

/-- The number of Ace cards Nell has now -/
def current_ace_cards : ℕ := remaining_baseball_cards + ace_baseball_difference

theorem nell_ace_cards : current_ace_cards = 376 := by
  sorry

end NUMINAMATH_CALUDE_nell_ace_cards_l1997_199744


namespace NUMINAMATH_CALUDE_fence_panels_count_l1997_199710

/-- Represents the components of a fence panel -/
structure FencePanel where
  sheets : Nat
  beams : Nat

/-- Represents the composition of sheets and beams -/
structure MetalComposition where
  rods_per_sheet : Nat
  rods_per_beam : Nat

/-- Calculates the number of fence panels given the total rods and composition -/
def calculate_fence_panels (total_rods : Nat) (panel : FencePanel) (comp : MetalComposition) : Nat :=
  total_rods / (panel.sheets * comp.rods_per_sheet + panel.beams * comp.rods_per_beam)

theorem fence_panels_count (total_rods : Nat) (panel : FencePanel) (comp : MetalComposition) :
  total_rods = 380 →
  panel.sheets = 3 →
  panel.beams = 2 →
  comp.rods_per_sheet = 10 →
  comp.rods_per_beam = 4 →
  calculate_fence_panels total_rods panel comp = 10 := by
  sorry

#eval calculate_fence_panels 380 ⟨3, 2⟩ ⟨10, 4⟩

end NUMINAMATH_CALUDE_fence_panels_count_l1997_199710


namespace NUMINAMATH_CALUDE_salary_sum_l1997_199737

/-- Given 5 individuals with an average salary and one known salary, 
    prove the sum of the other four salaries. -/
theorem salary_sum (average_salary : ℕ) (known_salary : ℕ) : 
  average_salary = 8800 → known_salary = 8000 → 
  (5 * average_salary) - known_salary = 36000 := by
  sorry

end NUMINAMATH_CALUDE_salary_sum_l1997_199737


namespace NUMINAMATH_CALUDE_tangent_intersection_of_specific_circles_l1997_199742

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The x-coordinate of the intersection point of a line tangent to two circles -/
def tangentIntersection (c1 c2 : Circle) : ℝ :=
  sorry

theorem tangent_intersection_of_specific_circles :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (18, 0), radius := 8 }
  tangentIntersection c1 c2 = 54 / 11 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_of_specific_circles_l1997_199742


namespace NUMINAMATH_CALUDE_find_y_value_l1997_199781

/-- Given five numbers in increasing order with specific conditions, prove y equals 16 -/
theorem find_y_value (x y : ℝ) : 
  2 < 5 ∧ 5 < x ∧ x < 10 ∧ 10 < y ∧  -- Increasing order condition
  x = 7 ∧  -- Median condition
  (2 + 5 + x + 10 + y) / 5 = 8  -- Mean condition
  → y = 16 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l1997_199781


namespace NUMINAMATH_CALUDE_license_plate_increase_l1997_199738

/-- The number of possible characters for letters in the new scheme -/
def new_letter_options : ℕ := 30

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_options : ℕ := 10

/-- The number of letters in the new license plate scheme -/
def new_letter_count : ℕ := 2

/-- The number of digits in the new license plate scheme -/
def new_digit_count : ℕ := 5

/-- The number of letters in the previous license plate scheme -/
def old_letter_count : ℕ := 3

/-- The number of digits in the previous license plate scheme -/
def old_digit_count : ℕ := 3

theorem license_plate_increase :
  (new_letter_options ^ new_letter_count * digit_options ^ new_digit_count) /
  (alphabet_size ^ old_letter_count * digit_options ^ old_digit_count) =
  (900 : ℚ) / 17576 * 100 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l1997_199738


namespace NUMINAMATH_CALUDE_no_solutions_to_equation_l1997_199762

theorem no_solutions_to_equation : 
  ¬∃ x : ℝ, (9 - x^2 ≥ 0) ∧ (Real.sqrt (9 - x^2) = x * Real.sqrt (9 - x^2) + x) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_to_equation_l1997_199762


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1997_199716

theorem sum_of_three_numbers (a b c : ℝ) : 
  a ≤ b → b ≤ c → b = 10 → (a + b + c) / 3 = a + 20 → (a + b + c) / 3 = c - 10 → 
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1997_199716


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_l1997_199711

theorem smallest_x_absolute_value (x : ℝ) : 
  (∀ y : ℝ, |5*y + 15| = 40 → y ≥ x) ↔ x = -11 ∧ |5*x + 15| = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_l1997_199711


namespace NUMINAMATH_CALUDE_distribution_theorem_l1997_199709

/-- The number of ways to distribute 6 volunteers into 4 groups and assign to 4 pavilions -/
def distribution_schemes : ℕ := 1080

/-- The number of volunteers -/
def num_volunteers : ℕ := 6

/-- The number of pavilions -/
def num_pavilions : ℕ := 4

/-- The number of groups with 2 people -/
def num_pairs : ℕ := 2

/-- The number of groups with 1 person -/
def num_singles : ℕ := 2

theorem distribution_theorem :
  (num_volunteers = 6) →
  (num_pavilions = 4) →
  (num_pairs = 2) →
  (num_singles = 2) →
  (num_pairs + num_singles = num_pavilions) →
  (2 * num_pairs + num_singles = num_volunteers) →
  distribution_schemes = 1080 := by
  sorry

#eval distribution_schemes

end NUMINAMATH_CALUDE_distribution_theorem_l1997_199709


namespace NUMINAMATH_CALUDE_quadratic_polynomial_root_l1997_199759

theorem quadratic_polynomial_root (x : ℂ) : 
  let p : ℂ → ℂ := λ z => 3 * z^2 - 12 * z + 24
  (p (2 + 2*I) = 0) ∧ (∀ z : ℂ, p z = 3 * z^2 + (-12 * z + 24)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_root_l1997_199759


namespace NUMINAMATH_CALUDE_fewer_spoons_purchased_l1997_199723

/-- The number of types of silverware --/
def numTypes : ℕ := 4

/-- The initially planned number of pieces per type --/
def initialPerType : ℕ := 15

/-- The total number of pieces actually purchased --/
def actualTotal : ℕ := 44

/-- Theorem stating that the number of fewer spoons purchased is 4 --/
theorem fewer_spoons_purchased :
  (numTypes * initialPerType - actualTotal) / numTypes = 4 := by
  sorry

end NUMINAMATH_CALUDE_fewer_spoons_purchased_l1997_199723


namespace NUMINAMATH_CALUDE_system_solutions_l1997_199769

theorem system_solutions :
  let S := {(x, y) : ℝ × ℝ | x^2 + y^2 = x ∧ 2*x*y = y}
  S = {(0, 0), (1, 0), (1/2, 1/2), (1/2, -1/2)} := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l1997_199769


namespace NUMINAMATH_CALUDE_lcm_of_24_36_40_l1997_199741

theorem lcm_of_24_36_40 : Nat.lcm (Nat.lcm 24 36) 40 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_24_36_40_l1997_199741


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l1997_199750

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem twentieth_term_of_sequence (a₁ d : ℤ) (h₁ : a₁ = 8) (h₂ : d = -3) :
  arithmeticSequenceTerm a₁ d 20 = -49 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l1997_199750


namespace NUMINAMATH_CALUDE_fitness_center_ratio_l1997_199743

theorem fitness_center_ratio (f m : ℕ) (h1 : f > 0) (h2 : m > 0) : 
  (45 * f + 25 * m) / (f + m) = 35 → f = m := by
  sorry

end NUMINAMATH_CALUDE_fitness_center_ratio_l1997_199743


namespace NUMINAMATH_CALUDE_base8_531_to_base7_l1997_199751

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def base10ToBase7 (n : ℕ) : List ℕ := sorry

/-- Checks if a list of digits represents a valid base 7 number --/
def isValidBase7 (digits : List ℕ) : Prop :=
  digits.all (· < 7)

theorem base8_531_to_base7 :
  let base10 := base8ToBase10 531
  let base7 := base10ToBase7 base10
  isValidBase7 base7 ∧ base7 = [1, 0, 0, 2] :=
by sorry

end NUMINAMATH_CALUDE_base8_531_to_base7_l1997_199751


namespace NUMINAMATH_CALUDE_max_value_when_xy_over_z_maximized_l1997_199733

/-- Given positive real numbers x, y, and z satisfying x^2 - 3xy + 4y^2 - z = 0,
    the maximum value of (2/x + 1/y - 2/z + 2) is 3 when xy/z is maximized. -/
theorem max_value_when_xy_over_z_maximized
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0) :
  let f := fun (x y z : ℝ) => 2/x + 1/y - 2/z + 2
  let g := fun (x y z : ℝ) => x*y/z
  ∃ (x' y' z' : ℝ),
    x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 ∧
    (∀ a b c, a > 0 → b > 0 → c > 0 → a^2 - 3*a*b + 4*b^2 - c = 0 →
      g a b c ≤ g x' y' z') ∧
    f x' y' z' = 3 ∧
    (∀ a b c, a > 0 → b > 0 → c > 0 → a^2 - 3*a*b + 4*b^2 - c = 0 →
      f a b c ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_when_xy_over_z_maximized_l1997_199733


namespace NUMINAMATH_CALUDE_count_negative_numbers_l1997_199717

theorem count_negative_numbers : let numbers := [-(-3), |-2|, (-2)^3, -3^2]
  ∃ (negative_count : ℕ), negative_count = (numbers.filter (λ x => x < 0)).length ∧ negative_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l1997_199717


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1997_199746

/-- A geometric sequence of positive integers with first term 3 and fourth term 243 has fifth term 243. -/
theorem geometric_sequence_fifth_term :
  ∀ (a : ℕ → ℕ),
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  (∀ n : ℕ, a n > 0) →  -- Positive integer condition
  a 1 = 3 →  -- First term is 3
  a 4 = 243 →  -- Fourth term is 243
  a 5 = 243 :=  -- Fifth term is 243
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1997_199746


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1997_199718

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 3 * i / (1 + i)
  Complex.im z = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1997_199718


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l1997_199730

theorem solve_quadratic_equation :
  ∃ x : ℚ, (10 - x)^2 = x^2 + 4 ∧ x = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l1997_199730


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l1997_199784

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + (k - 3) = 0) →
  k ≥ 3/4 ∧ k ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l1997_199784


namespace NUMINAMATH_CALUDE_distance_calculation_l1997_199758

theorem distance_calculation (speed_to : ℝ) (speed_from : ℝ) (total_time : ℝ) 
  (h1 : speed_to = 50)
  (h2 : speed_from = 75)
  (h3 : total_time = 10) :
  (total_time * speed_to * speed_from) / (speed_to + speed_from) = 300 :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l1997_199758


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1997_199704

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x*y = 2) :
  x + y ≥ 2 * Real.sqrt 3 - 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1997_199704


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l1997_199785

-- Define the purchase price
def purchase_price : ℝ := 50

-- Define the linear relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -20 * x + 2600

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - purchase_price) * sales_volume x

-- Define the constraint that selling price is not lower than purchase price
def price_constraint (x : ℝ) : Prop := x ≥ purchase_price

-- Define the constraint that profit per shirt should not exceed 30% of purchase price
def profit_constraint (x : ℝ) : Prop := x - purchase_price ≤ 0.3 * purchase_price

-- Theorem statement
theorem optimal_price_and_profit :
  ∃ (x : ℝ), 
    price_constraint x ∧ 
    profit_constraint x ∧ 
    (∀ y : ℝ, price_constraint y → profit_constraint y → profit x ≥ profit y) ∧
    x = 65 ∧
    profit x = 19500 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_and_profit_l1997_199785


namespace NUMINAMATH_CALUDE_product_of_sqrt5_plus_minus_2_l1997_199736

theorem product_of_sqrt5_plus_minus_2 :
  let a := Real.sqrt 5 + 2
  let b := Real.sqrt 5 - 2
  a * b = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_sqrt5_plus_minus_2_l1997_199736


namespace NUMINAMATH_CALUDE_election_win_margin_l1997_199779

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) (winner_percentage : ℚ) :
  winner_percentage = 62 / 100 →
  winner_votes = 930 →
  winner_votes = total_votes * winner_percentage →
  winner_votes - (total_votes - winner_votes) = 360 :=
by sorry

end NUMINAMATH_CALUDE_election_win_margin_l1997_199779


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2014_l1997_199734

theorem no_natural_square_diff_2014 : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2014_l1997_199734


namespace NUMINAMATH_CALUDE_shooting_test_probability_l1997_199745

/-- The probability of a successful shot -/
def p : ℚ := 2/3

/-- The number of successful shots required to pass -/
def required_successes : ℕ := 3

/-- The maximum number of shots allowed -/
def max_shots : ℕ := 5

/-- The probability of passing the shooting test -/
def pass_probability : ℚ := 64/81

theorem shooting_test_probability :
  (p^required_successes) +
  (Nat.choose 4 required_successes * p^required_successes * (1-p)) +
  (Nat.choose 5 required_successes * p^required_successes * (1-p)^2) = pass_probability :=
sorry

end NUMINAMATH_CALUDE_shooting_test_probability_l1997_199745


namespace NUMINAMATH_CALUDE_abs_neg_nine_l1997_199753

theorem abs_neg_nine : |(-9 : ℤ)| = 9 := by sorry

end NUMINAMATH_CALUDE_abs_neg_nine_l1997_199753


namespace NUMINAMATH_CALUDE_loan_interest_difference_l1997_199748

/-- Proves that for a loan of 2000 at 3% simple interest for 3 years, 
    the difference between the principal and the interest is 1940 -/
theorem loan_interest_difference : 
  let principal : ℚ := 2000
  let rate : ℚ := 3 / 100
  let time : ℚ := 3
  let interest := principal * rate * time
  principal - interest = 1940 := by sorry

end NUMINAMATH_CALUDE_loan_interest_difference_l1997_199748


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1997_199780

/-- The quadratic equation (k-1)x^2 - 2kx + k + 3 = 0 has real roots if and only if k ≤ 3/2 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + k + 3 = 0) ↔ k ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1997_199780


namespace NUMINAMATH_CALUDE_total_games_attended_l1997_199700

def games_this_year : ℕ := 15
def games_last_year : ℕ := 39

theorem total_games_attended : games_this_year + games_last_year = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_games_attended_l1997_199700


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l1997_199705

theorem largest_divisor_of_consecutive_odd_product :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧
  (∀ k : ℕ, k > m → ¬(k ∣ ((2*n - 1) * (2*n + 1) * (2*n + 3)))) ∧
  (3 ∣ ((2*n - 1) * (2*n + 1) * (2*n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l1997_199705


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1997_199714

-- Problem 1
theorem problem_one : -2^2 - |2 - 5| + (-1) * 2 = -1 := by sorry

-- Problem 2
theorem problem_two : ∃! x : ℝ, 5 * x - 2 = 3 * x + 18 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1997_199714


namespace NUMINAMATH_CALUDE_noodles_already_have_l1997_199768

/-- The amount of beef Tom has in pounds -/
def beef : ℕ := 10

/-- The ratio of noodles to beef -/
def noodle_to_beef_ratio : ℕ := 2

/-- The weight of each noodle package in pounds -/
def package_weight : ℕ := 2

/-- The number of packages Tom needs to buy -/
def packages_to_buy : ℕ := 8

/-- The total amount of noodles needed in pounds -/
def total_noodles_needed : ℕ := noodle_to_beef_ratio * beef

/-- The amount of noodles Tom needs to buy in pounds -/
def noodles_to_buy : ℕ := packages_to_buy * package_weight

theorem noodles_already_have : 
  total_noodles_needed - noodles_to_buy = 4 := by
  sorry

end NUMINAMATH_CALUDE_noodles_already_have_l1997_199768


namespace NUMINAMATH_CALUDE_initial_children_meals_l1997_199703

/-- Calculates the number of meals initially available for children given the total adult meals and remaining meals after some adults eat. -/
def children_meals (total_adult_meals : ℕ) (adults_eaten : ℕ) (remaining_child_meals : ℕ) : ℕ :=
  (total_adult_meals * remaining_child_meals) / (total_adult_meals - adults_eaten)

/-- Proves that the number of meals initially available for children is 90. -/
theorem initial_children_meals :
  children_meals 70 14 72 = 90 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_meals_l1997_199703


namespace NUMINAMATH_CALUDE_solution_relationship_l1997_199727

theorem solution_relationship (x y : ℝ) : 
  2 * x + y = 7 → x - y = 5 → x + 2 * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_relationship_l1997_199727


namespace NUMINAMATH_CALUDE_croissant_resting_time_l1997_199707

theorem croissant_resting_time (fold_count : ℕ) (fold_time : ℕ) (mixing_time : ℕ) (baking_time : ℕ) (total_time : ℕ) :
  fold_count = 4 →
  fold_time = 5 →
  mixing_time = 10 →
  baking_time = 30 →
  total_time = 6 * 60 →
  (total_time - (mixing_time + fold_count * fold_time + baking_time)) / fold_count = 75 := by
  sorry

end NUMINAMATH_CALUDE_croissant_resting_time_l1997_199707


namespace NUMINAMATH_CALUDE_divisible_by_2000_arrangement_l1997_199724

theorem divisible_by_2000_arrangement (nums : List ℕ) (h : nums.length = 23) :
  ∃ (arrangement : List ℕ → ℕ), arrangement nums % 2000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_2000_arrangement_l1997_199724


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_18_l1997_199767

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Primality test for natural numbers -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 819 is the smallest prime number whose digits sum to 18 -/
theorem smallest_prime_digit_sum_18 : 
  (is_prime 819 ∧ digit_sum 819 = 18) ∧ 
  ∀ n : ℕ, n < 819 → ¬(is_prime n ∧ digit_sum n = 18) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_18_l1997_199767


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1997_199798

def constant_term (n : ℕ) : ℕ :=
  Nat.choose n 0 + 
  Nat.choose n 2 * Nat.choose 2 1 + 
  Nat.choose n 4 * Nat.choose 4 2 + 
  Nat.choose n 6 * Nat.choose 6 3

theorem constant_term_expansion : constant_term 6 = 141 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1997_199798


namespace NUMINAMATH_CALUDE_equation_solution_l1997_199721

theorem equation_solution : ∃ (x : ℝ), 
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6 ∧
  (x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2) ∧
  (3 * x + 6) / (x^2 + 5*x - 6) = (x - 3) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1997_199721


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l1997_199715

theorem unique_solution_inequality (x : ℝ) :
  x > 0 →
  16 - x ≥ 0 →
  16 * x - x^3 ≥ 0 →
  x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l1997_199715


namespace NUMINAMATH_CALUDE_circle_radius_equality_l1997_199765

theorem circle_radius_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 37) (h₂ : r₂ = 23) :
  ∃ r : ℝ, r^2 = (r₁^2 - r₂^2) ∧ r = 2 * Real.sqrt 210 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_equality_l1997_199765


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1997_199712

theorem division_remainder_proof (divisor quotient dividend remainder : ℕ) : 
  divisor = 21 →
  quotient = 14 →
  dividend = 301 →
  dividend = divisor * quotient + remainder →
  remainder = 7 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1997_199712


namespace NUMINAMATH_CALUDE_circle_intersection_ratio_l1997_199776

theorem circle_intersection_ratio (m : ℝ) (h : 0 < m ∧ m < 1) :
  let R : ℝ := 1  -- We can set R = 1 without loss of generality
  let common_area := 2 * R^2 * (Real.arccos m - m * Real.sqrt (1 - m^2))
  let third_circle_area := π * (m * R)^2
  common_area / third_circle_area = 2 * (Real.arccos m - m * Real.sqrt (1 - m^2)) / (π * m^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_ratio_l1997_199776


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l1997_199782

theorem least_reducible_fraction : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 0 < m ∧ m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 13) ∧ k ∣ (5*m + 6))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 13) ∧ k ∣ (5*n + 6)) ∧
  n = 84 :=
by sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l1997_199782


namespace NUMINAMATH_CALUDE_ellipse_minimum_area_l1997_199788

/-- An ellipse containing two specific circles has a minimum area of 16π -/
theorem ellipse_minimum_area :
  ∀ a b : ℝ,
  (∀ x y : ℝ, x^2 / (4*a^2) + y^2 / (4*b^2) = 1 →
    ((x - 2)^2 + y^2 ≤ 4 ∧ (x + 2)^2 + y^2 ≤ 4)) →
  4 * π * a * b ≥ 16 * π :=
by sorry

end NUMINAMATH_CALUDE_ellipse_minimum_area_l1997_199788


namespace NUMINAMATH_CALUDE_divisibility_implies_sum_divisibility_l1997_199726

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem divisibility_implies_sum_divisibility (n : ℕ) 
  (h1 : n < 10000) (h2 : n % 99 = 0) : 
  (sum_of_digits n) % 18 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_sum_divisibility_l1997_199726


namespace NUMINAMATH_CALUDE_bakery_sales_percentage_l1997_199774

theorem bakery_sales_percentage (cake_percent cookie_percent : ℝ) 
  (h_cake : cake_percent = 42)
  (h_cookie : cookie_percent = 25) :
  100 - (cake_percent + cookie_percent) = 33 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sales_percentage_l1997_199774


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l1997_199791

/-- Given 3 bugs, each eating 2 flowers, prove that the total number of flowers eaten is 6. -/
theorem bugs_eating_flowers : 
  let num_bugs : ℕ := 3
  let flowers_per_bug : ℕ := 2
  num_bugs * flowers_per_bug = 6 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l1997_199791
