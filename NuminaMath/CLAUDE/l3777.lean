import Mathlib

namespace NUMINAMATH_CALUDE_total_crackers_bought_l3777_377744

/-- The number of boxes of crackers Darren bought -/
def darren_boxes : ℕ := 4

/-- The number of crackers in each box -/
def crackers_per_box : ℕ := 24

/-- The number of boxes Calvin bought -/
def calvin_boxes : ℕ := 2 * darren_boxes - 1

/-- The total number of crackers bought by both Darren and Calvin -/
def total_crackers : ℕ := darren_boxes * crackers_per_box + calvin_boxes * crackers_per_box

theorem total_crackers_bought :
  total_crackers = 264 := by
  sorry

end NUMINAMATH_CALUDE_total_crackers_bought_l3777_377744


namespace NUMINAMATH_CALUDE_coefficient_x2y2_is_70_l3777_377781

/-- The coefficient of x^2 * y^2 in the expansion of (x/√y - y/√x)^8 -/
def coefficient_x2y2 : ℕ :=
  let expression := (fun x y => x / Real.sqrt y - y / Real.sqrt x) ^ 8
  70  -- Placeholder for the actual coefficient

/-- The coefficient of x^2 * y^2 in the expansion of (x/√y - y/√x)^8 is 70 -/
theorem coefficient_x2y2_is_70 : coefficient_x2y2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_is_70_l3777_377781


namespace NUMINAMATH_CALUDE_unique_determination_from_subset_sums_l3777_377761

/-- Given a set of n integers, this function returns all possible subset sums excluding the empty subset -/
def allSubsetSums (s : Finset Int) : Finset Int :=
  sorry

theorem unique_determination_from_subset_sums
  (n : Nat)
  (s : Finset Int)
  (h1 : s.card = n)
  (h2 : 0 ∉ allSubsetSums s)
  (h3 : (allSubsetSums s).card = 2^n - 1) :
  ∀ t : Finset Int, allSubsetSums s = allSubsetSums t → s = t :=
sorry

end NUMINAMATH_CALUDE_unique_determination_from_subset_sums_l3777_377761


namespace NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l3777_377791

theorem smallest_n_for_quadruplets : ∃ (n : ℕ+), 
  (∃! (quad_count : ℕ), quad_count = 154000 ∧ 
    (∃ (S : Finset (ℕ+ × ℕ+ × ℕ+ × ℕ+)), 
      Finset.card S = quad_count ∧
      ∀ (a b c d : ℕ+), (a, b, c, d) ∈ S ↔ 
        (Nat.gcd a.val (Nat.gcd b.val (Nat.gcd c.val d.val)) = 154 ∧
         Nat.lcm a.val (Nat.lcm b.val (Nat.lcm c.val d.val)) = n.val))) ∧
  (∀ (m : ℕ+), m < n →
    ¬∃ (quad_count : ℕ), quad_count = 154000 ∧
      (∃ (S : Finset (ℕ+ × ℕ+ × ℕ+ × ℕ+)),
        Finset.card S = quad_count ∧
        ∀ (a b c d : ℕ+), (a, b, c, d) ∈ S ↔
          (Nat.gcd a.val (Nat.gcd b.val (Nat.gcd c.val d.val)) = 154 ∧
           Nat.lcm a.val (Nat.lcm b.val (Nat.lcm c.val d.val)) = m.val))) ∧
  n = 25520328 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l3777_377791


namespace NUMINAMATH_CALUDE_xy_inequality_l3777_377750

theorem xy_inequality (x y : ℝ) (h : x^2 + y^2 - x*y = 1) :
  (-2 : ℝ) ≤ x + y ∧ x + y ≤ 2 ∧ 2/3 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l3777_377750


namespace NUMINAMATH_CALUDE_negation_relationship_l3777_377789

theorem negation_relationship (x : ℝ) : 
  (¬(0 < x ∧ x < 2) → ¬(1/x ≥ 1)) ∧ ¬(¬(1/x ≥ 1) → ¬(0 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_relationship_l3777_377789


namespace NUMINAMATH_CALUDE_factor_implies_k_value_l3777_377753

/-- Given a quadratic trinomial 2x^2 + 3x - k with a factor (2x - 5), k equals 20 -/
theorem factor_implies_k_value (k : ℝ) : 
  (∃ (q : ℝ → ℝ), ∀ x, 2*x^2 + 3*x - k = (2*x - 5) * q x) → 
  k = 20 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_k_value_l3777_377753


namespace NUMINAMATH_CALUDE_proportional_function_slope_l3777_377794

/-- A proportional function passing through the point (3, -5) has a slope of -5/3 -/
theorem proportional_function_slope (k : ℝ) (h1 : k ≠ 0) 
  (h2 : -5 = k * 3) : k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_slope_l3777_377794


namespace NUMINAMATH_CALUDE_nick_running_speed_l3777_377797

/-- Represents the speed required for the fourth lap to achieve a target average speed -/
def fourth_lap_speed (first_three_speed : ℝ) (target_avg_speed : ℝ) : ℝ :=
  4 * target_avg_speed - 3 * first_three_speed

/-- Proves that if a runner completes three laps at 9 mph and needs to achieve an average 
    speed of 10 mph for four laps, then the speed required for the fourth lap is 15 mph -/
theorem nick_running_speed : fourth_lap_speed 9 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_nick_running_speed_l3777_377797


namespace NUMINAMATH_CALUDE_max_diff_inequality_l3777_377775

open Function Set

variable {n : ℕ}

/-- Two strictly increasing finite sequences of real numbers -/
def StrictlyIncreasingSeq (a b : Fin n → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j ∧ b i < b j

theorem max_diff_inequality
  (a b : Fin n → ℝ)
  (h_inc : StrictlyIncreasingSeq a b)
  (f : Fin n → Fin n)
  (h_bij : Bijective f) :
  (⨆ i, |a i - b i|) ≤ (⨆ i, |a i - b (f i)|) :=
sorry

end NUMINAMATH_CALUDE_max_diff_inequality_l3777_377775


namespace NUMINAMATH_CALUDE_smallest_square_with_specific_digits_l3777_377711

theorem smallest_square_with_specific_digits : 
  let n : ℕ := 666667
  ∀ m : ℕ, m < n → 
    (m ^ 2 < 444445 * 10^6) ∨ 
    (m ^ 2 ≥ 444446 * 10^6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_with_specific_digits_l3777_377711


namespace NUMINAMATH_CALUDE_donuts_left_for_coworkers_l3777_377706

def total_donuts : ℕ := 30
def gluten_free_donuts : ℕ := 12
def regular_donuts : ℕ := total_donuts - gluten_free_donuts

def gluten_free_eaten_driving : ℕ := 1
def regular_eaten_driving : ℕ := 0

def gluten_free_afternoon_snack : ℕ := 2
def regular_afternoon_snack : ℕ := 4

theorem donuts_left_for_coworkers :
  total_donuts - 
  (gluten_free_eaten_driving + regular_eaten_driving + 
   gluten_free_afternoon_snack + regular_afternoon_snack) = 23 := by
  sorry

end NUMINAMATH_CALUDE_donuts_left_for_coworkers_l3777_377706


namespace NUMINAMATH_CALUDE_evaluate_expression_l3777_377708

theorem evaluate_expression : -(18 / 3 * 8 - 40 + 5^2) = -33 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3777_377708


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3777_377718

/-- Given a line L1 with equation x - 2y + 3 = 0 and a point A(1, 2),
    the line L2 passing through A and perpendicular to L1 has the equation 2x + y - 4 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) : 
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + 3 = 0
  let A : ℝ × ℝ := (1, 2)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 4 = 0
  (∀ x y, L1 x y ↔ x - 2*y + 3 = 0) →
  (L2 A.1 A.2) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) * ((x₂ - x₁) * (x₂ - x₁)) = 0) →
  ∀ x y, L2 x y ↔ 2*x + y - 4 = 0 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l3777_377718


namespace NUMINAMATH_CALUDE_debate_team_girls_l3777_377766

theorem debate_team_girls (boys : ℕ) (groups : ℕ) (members_per_group : ℕ) : 
  boys = 26 → groups = 8 → members_per_group = 9 → 
  (groups * members_per_group) - boys = 46 := by sorry

end NUMINAMATH_CALUDE_debate_team_girls_l3777_377766


namespace NUMINAMATH_CALUDE_opposite_grey_is_violet_l3777_377733

-- Define the colors
inductive Color
| Yellow
| Grey
| Orange
| Violet
| Blue
| Black

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : Fin 6 → Face

-- Define a view of the cube
structure View where
  top : Color
  front : Color
  right : Color

-- Define the given views
def view1 : View := { top := Color.Yellow, front := Color.Blue, right := Color.Black }
def view2 : View := { top := Color.Orange, front := Color.Yellow, right := Color.Black }
def view3 : View := { top := Color.Orange, front := Color.Violet, right := Color.Black }

-- Theorem statement
theorem opposite_grey_is_violet (c : Cube) 
  (h1 : ∃ (f1 f2 f3 : Fin 6), c.faces f1 = { color := view1.top } ∧ 
                               c.faces f2 = { color := view1.front } ∧ 
                               c.faces f3 = { color := view1.right })
  (h2 : ∃ (f1 f2 f3 : Fin 6), c.faces f1 = { color := view2.top } ∧ 
                               c.faces f2 = { color := view2.front } ∧ 
                               c.faces f3 = { color := view2.right })
  (h3 : ∃ (f1 f2 f3 : Fin 6), c.faces f1 = { color := view3.top } ∧ 
                               c.faces f2 = { color := view3.front } ∧ 
                               c.faces f3 = { color := view3.right })
  (h4 : ∃! (f : Fin 6), c.faces f = { color := Color.Grey }) :
  ∃ (f1 f2 : Fin 6), c.faces f1 = { color := Color.Grey } ∧ 
                     c.faces f2 = { color := Color.Violet } ∧ 
                     f1 ≠ f2 ∧ 
                     ∀ (f3 : Fin 6), f3 ≠ f1 ∧ f3 ≠ f2 → 
                       (c.faces f3).color ≠ Color.Grey ∧ (c.faces f3).color ≠ Color.Violet :=
by
  sorry


end NUMINAMATH_CALUDE_opposite_grey_is_violet_l3777_377733


namespace NUMINAMATH_CALUDE_x_fourth_minus_four_x_cubed_plus_four_x_squared_plus_four_equals_five_l3777_377720

theorem x_fourth_minus_four_x_cubed_plus_four_x_squared_plus_four_equals_five :
  ∀ x : ℝ, x = 1 + Real.sqrt 2 → x^4 - 4*x^3 + 4*x^2 + 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_four_x_cubed_plus_four_x_squared_plus_four_equals_five_l3777_377720


namespace NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l3777_377763

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Theorem stating that "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_neither_sufficient_nor_necessary :
  (∃ (a : ℕ → ℝ) (q : ℝ), q > 1 ∧ GeometricSequence a q ∧ ¬IncreasingSequence a) ∧
  (∃ (a : ℕ → ℝ) (q : ℝ), q ≤ 1 ∧ GeometricSequence a q ∧ IncreasingSequence a) :=
sorry

end NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l3777_377763


namespace NUMINAMATH_CALUDE_can_identify_counterfeit_coins_l3777_377736

/-- Represents the result of checking a pair of coins -/
inductive CheckResult
  | Zero
  | One
  | Two

/-- Represents a coin -/
inductive Coin
  | One
  | Two
  | Three
  | Four
  | Five

/-- A function that checks a pair of coins and returns the number of counterfeit coins -/
def checkPair (c1 c2 : Coin) : CheckResult := sorry

/-- The set of all coins -/
def allCoins : Finset Coin := sorry

/-- The set of counterfeit coins -/
def counterfeitCoins : Finset Coin := sorry

/-- The four pairs of coins to be checked -/
def pairsToCheck : List (Coin × Coin) := sorry

theorem can_identify_counterfeit_coins :
  (Finset.card allCoins = 5) →
  (Finset.card counterfeitCoins = 2) →
  (List.length pairsToCheck = 4) →
  ∃ (f : List CheckResult → Finset Coin),
    ∀ (results : List CheckResult),
      List.length results = 4 →
      results = List.map (fun (p : Coin × Coin) => checkPair p.1 p.2) pairsToCheck →
      f results = counterfeitCoins :=
sorry

end NUMINAMATH_CALUDE_can_identify_counterfeit_coins_l3777_377736


namespace NUMINAMATH_CALUDE_leadership_choices_l3777_377745

/-- The number of ways to choose leadership in an organization --/
def choose_leadership (total_members : ℕ) (president_count : ℕ) (vp_count : ℕ) (managers_per_vp : ℕ) : ℕ :=
  let remaining_after_president := total_members - president_count
  let remaining_after_vps := remaining_after_president - vp_count
  let remaining_after_vp1_managers := remaining_after_vps - managers_per_vp
  total_members *
  remaining_after_president *
  (remaining_after_president - 1) *
  (Nat.choose remaining_after_vps managers_per_vp) *
  (Nat.choose remaining_after_vp1_managers managers_per_vp)

/-- Theorem stating the number of ways to choose leadership in the given organization --/
theorem leadership_choices :
  choose_leadership 12 1 2 2 = 554400 :=
by sorry

end NUMINAMATH_CALUDE_leadership_choices_l3777_377745


namespace NUMINAMATH_CALUDE_logarithm_product_identity_l3777_377758

theorem logarithm_product_identity (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy1 : y ≠ 1) :
  Real.log x ^ 2 / Real.log (y ^ 3) *
  Real.log (y ^ 3) / Real.log (x ^ 4) *
  Real.log (x ^ 4) / Real.log (y ^ 5) *
  Real.log (y ^ 5) / Real.log (x ^ 2) =
  Real.log x / Real.log y := by
  sorry

end NUMINAMATH_CALUDE_logarithm_product_identity_l3777_377758


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3777_377709

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 240 →
  crossing_time = 20 →
  train_speed_kmh = 70.2 →
  ∃ (bridge_length : ℝ), bridge_length = 150 := by
    sorry


end NUMINAMATH_CALUDE_bridge_length_calculation_l3777_377709


namespace NUMINAMATH_CALUDE_intersection_point_B_coords_l3777_377705

/-- Two circles with centers on the line y = 1 - x, intersecting at points A and B -/
structure IntersectingCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  centers_on_line : ∀ c, c = O₁ ∨ c = O₂ → c.2 = 1 - c.1
  A_coords : A = (-7, 9)
  intersection : A ≠ B

/-- The theorem stating that point B has coordinates (-8, 8) -/
theorem intersection_point_B_coords (circles : IntersectingCircles) : circles.B = (-8, 8) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_B_coords_l3777_377705


namespace NUMINAMATH_CALUDE_minAbsNegativeFractions_minAbsNegativeTwo_solveMinAbsEquation_l3777_377755

-- Define the min|a,b| operation for rational numbers
def minAbs (a b : ℚ) : ℚ := min a b

-- Theorem 1
theorem minAbsNegativeFractions : minAbs (-5/2) (-4/3) = -5/2 := by sorry

-- Theorem 2
theorem minAbsNegativeTwo (y : ℚ) (h : y < -2) : minAbs (-2) y = y := by sorry

-- Theorem 3
theorem solveMinAbsEquation : 
  ∃ x : ℚ, (minAbs (-x) 0 = -5 + 2*x) ∧ (x = 5/3) := by sorry

end NUMINAMATH_CALUDE_minAbsNegativeFractions_minAbsNegativeTwo_solveMinAbsEquation_l3777_377755


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3777_377730

theorem polynomial_factorization (x : ℝ) : x^2 - x = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3777_377730


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3777_377700

theorem min_value_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) :
  (1 / (2*a + 3*b) + 1 / (2*b + 3*c) + 1 / (2*c + 3*a)) ≥ 9/5 ∧
  (1 / (2*a + 3*b) + 1 / (2*b + 3*c) + 1 / (2*c + 3*a) = 9/5 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3777_377700


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3777_377776

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ k : ℕ+, k < n → (7 : ℤ)^(k : ℕ) % 5 ≠ (k : ℤ)^7 % 5) ∧ 
  (7 : ℤ)^(n : ℕ) % 5 = (n : ℤ)^7 % 5 → 
  n = 7 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3777_377776


namespace NUMINAMATH_CALUDE_geometric_sequence_pairs_l3777_377743

/-- The number of ordered pairs (a, r) satisfying the given conditions -/
def num_pairs : ℕ := 26^3

/-- The base of the logarithm and the exponent in the final equation -/
def base : ℕ := 2015
def exponent : ℕ := 155

theorem geometric_sequence_pairs :
  ∃ (S : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ S ↔ 
      let (a, r) := p
      (a > 0 ∧ r > 0) ∧ (a * r^6 = base^exponent)) ∧
    Finset.card S = num_pairs :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_pairs_l3777_377743


namespace NUMINAMATH_CALUDE_correct_selection_count_l3777_377756

/-- Represents a basketball team with twins -/
structure BasketballTeam where
  total_players : Nat
  twin_sets : Nat
  non_twins : Nat

/-- Calculates the number of ways to select players for a game -/
def select_players (team : BasketballTeam) (to_select : Nat) : Nat :=
  sorry

/-- The specific basketball team from the problem -/
def our_team : BasketballTeam := {
  total_players := 16,
  twin_sets := 3,
  non_twins := 10
}

/-- Theorem stating the correct number of ways to select players -/
theorem correct_selection_count :
  select_players our_team 7 = 1380 := by sorry

end NUMINAMATH_CALUDE_correct_selection_count_l3777_377756


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3777_377739

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the perpendicular bisector of a line with slope m
def perpendicular_bisector (m : ℝ) (x y : ℝ) : Prop := x = -1/m * y + (2*m^2 + 3)

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

theorem parabola_intersection_theorem (m : ℝ) :
  ∃ (xA yA xB yB xC yC xD yD : ℝ),
    -- A and B are on the parabola and the line through focus
    parabola xA yA ∧ parabola xB yB ∧
    line_through_focus m xA yA ∧ line_through_focus m xB yB ∧
    -- C and D are on the parabola and the perpendicular bisector
    parabola xC yC ∧ parabola xD yD ∧
    perpendicular_bisector m xC yC ∧ perpendicular_bisector m xD yD ∧
    -- AC is perpendicular to AD
    dot_product (xC - xA) (yC - yA) (xD - xA) (yD - yA) = 0 →
    m = 1 ∨ m = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3777_377739


namespace NUMINAMATH_CALUDE_distribute_6_3_l3777_377786

/-- The number of ways to distribute n identical objects into k distinct containers,
    with each container having at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The problem statement -/
theorem distribute_6_3 : distribute 6 3 = 10 := by sorry

end NUMINAMATH_CALUDE_distribute_6_3_l3777_377786


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l3777_377790

/-- Represents the rates for running, bicycling, and roller-skating -/
structure Rates where
  running : ℕ
  bicycling : ℕ
  roller_skating : ℕ

/-- Tom's total distance -/
def tom_distance (r : Rates) : ℕ := 3 * r.running + 4 * r.bicycling + 2 * r.roller_skating

/-- Jerry's total distance -/
def jerry_distance (r : Rates) : ℕ := 3 * r.running + 6 * r.bicycling + 2 * r.roller_skating

/-- Sum of squares of rates -/
def sum_of_squares (r : Rates) : ℕ := r.running^2 + r.bicycling^2 + r.roller_skating^2

theorem rates_sum_of_squares :
  ∃ r : Rates,
    tom_distance r = 104 ∧
    jerry_distance r = 140 ∧
    sum_of_squares r = 440 := by
  sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l3777_377790


namespace NUMINAMATH_CALUDE_circle_radius_from_arc_and_angle_l3777_377760

/-- 
Given a circle where an arc of length 5π cm corresponds to a central angle of 150°, 
the radius of the circle is 6 cm.
-/
theorem circle_radius_from_arc_and_angle : 
  ∀ (r : ℝ), 
  (150 / 180 : ℝ) * Real.pi * r = 5 * Real.pi → 
  r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_arc_and_angle_l3777_377760


namespace NUMINAMATH_CALUDE_difference_of_squares_l3777_377780

theorem difference_of_squares : 550^2 - 450^2 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3777_377780


namespace NUMINAMATH_CALUDE_min_area_is_three_l3777_377717

/-- Triangle ABC with A at origin, B at (30, 18), and C with integer coordinates -/
structure Triangle :=
  (c_x : ℤ)
  (c_y : ℤ)

/-- Area of triangle ABC given coordinates of C -/
def area (t : Triangle) : ℚ :=
  (1 / 2 : ℚ) * |30 * t.c_y - 18 * t.c_x|

/-- The minimum area of triangle ABC is 3 -/
theorem min_area_is_three :
  ∃ (t : Triangle), area t = 3 ∧ ∀ (t' : Triangle), area t' ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_area_is_three_l3777_377717


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3777_377715

theorem point_in_fourth_quadrant :
  let P : ℝ × ℝ := (Real.tan (549 * π / 180), Real.cos (549 * π / 180))
  (P.1 > 0) ∧ (P.2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3777_377715


namespace NUMINAMATH_CALUDE_sarahs_age_proof_l3777_377782

theorem sarahs_age_proof (ana billy mark sarah : ℕ) : 
  ana + 8 = 40 → 
  billy = ana / 2 → 
  mark = billy + 4 → 
  sarah = 3 * mark - 4 → 
  sarah = 56 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_age_proof_l3777_377782


namespace NUMINAMATH_CALUDE_women_handshakes_fifteen_couples_l3777_377710

/-- The number of handshakes among women in a group of married couples -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 15 married couples, if only women shake hands with other women
    (excluding their spouses), the total number of handshakes is 105. -/
theorem women_handshakes_fifteen_couples :
  handshakes 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_women_handshakes_fifteen_couples_l3777_377710


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l3777_377734

/-- The area of a right-angled triangle inscribed in a circle of radius 100, 
    with acute angles α and β satisfying tan α = 4 tan β, is equal to 8000. -/
theorem inscribed_triangle_area (α β : Real) (h1 : α > 0) (h2 : β > 0) (h3 : α + β = Real.pi / 2) 
  (h4 : Real.tan α = 4 * Real.tan β) : 
  let r : Real := 100
  let area := r^2 * Real.sin α * Real.sin β
  area = 8000 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l3777_377734


namespace NUMINAMATH_CALUDE_igor_process_terminates_l3777_377770

/-- Appends a digit to make a number divisible by 11 -/
def appendDigit (n : Nat) : Nat :=
  let m := n * 10
  (m + (11 - m % 11) % 11)

/-- Performs one step of Igor's process -/
def igorStep (n : Nat) : Nat :=
  (appendDigit n) / 11

/-- Checks if Igor can continue the process -/
def canContinue (n : Nat) : Bool :=
  ∃ (d : Nat), d < 10 ∧ (n * 10 + d) % 11 = 0

/-- The sequence of numbers generated by Igor's process -/
def igorSequence : Nat → Nat
  | 0 => 2018
  | n + 1 => igorStep (igorSequence n)

theorem igor_process_terminates :
  ∃ (N : Nat), ¬(canContinue (igorSequence N)) :=
sorry

end NUMINAMATH_CALUDE_igor_process_terminates_l3777_377770


namespace NUMINAMATH_CALUDE_system_solution_l3777_377768

theorem system_solution : ∃! (x y z : ℝ), 
  (x - y ≥ z ∧ x^2 + 4*y^2 + 5 = 4*z) ∧ 
  x = 2 ∧ y = -1/2 ∧ z = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3777_377768


namespace NUMINAMATH_CALUDE_crayons_lost_theorem_l3777_377702

/-- The number of crayons lost or given away -/
def crayons_lost_or_given_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that the number of crayons lost or given away is correct -/
theorem crayons_lost_theorem (initial : ℕ) (remaining : ℕ) 
  (h : initial ≥ remaining) : 
  crayons_lost_or_given_away initial remaining = initial - remaining :=
by
  sorry

#eval crayons_lost_or_given_away 479 134

end NUMINAMATH_CALUDE_crayons_lost_theorem_l3777_377702


namespace NUMINAMATH_CALUDE_min_value_expression_l3777_377762

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 8 ∧
  ((x^3 / (y - 1)) + (y^3 / (x - 1)) = 8 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3777_377762


namespace NUMINAMATH_CALUDE_linear_function_proof_l3777_377726

/-- A linear function passing through (0,5) and parallel to y=x -/
def f (x : ℝ) : ℝ := x + 5

theorem linear_function_proof :
  (f 0 = 5) ∧ 
  (∀ x y : ℝ, f (x + y) - f x = y) ∧
  (∀ x : ℝ, f x = x + 5) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l3777_377726


namespace NUMINAMATH_CALUDE_customers_who_left_l3777_377777

-- Define the initial number of customers
def initial_customers : ℕ := 13

-- Define the number of new customers
def new_customers : ℕ := 4

-- Define the final number of customers
def final_customers : ℕ := 9

-- Theorem to prove the number of customers who left
theorem customers_who_left :
  ∃ (left : ℕ), initial_customers - left + new_customers = final_customers ∧ left = 8 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_left_l3777_377777


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3777_377752

theorem arithmetic_geometric_sequence_sum (a b c d : ℝ) : 
  (∃ k : ℝ, a = 6 + k ∧ b = 6 + 2*k ∧ 48 = 6 + 3*k) →  -- arithmetic sequence condition
  (∃ q : ℝ, c = 6*q ∧ d = 6*q^2 ∧ 48 = 6*q^3) →        -- geometric sequence condition
  a + b + c + d = 111 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3777_377752


namespace NUMINAMATH_CALUDE_remainder_theorem_l3777_377748

def polynomial (x : ℝ) : ℝ := 8*x^4 - 6*x^3 + 17*x^2 - 27*x + 35

def divisor (x : ℝ) : ℝ := 2*x - 8

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (divisor x) * q x + 1863 :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3777_377748


namespace NUMINAMATH_CALUDE_school_children_count_l3777_377784

/-- Proves that the number of children in a school is 780, given the conditions of banana distribution -/
theorem school_children_count : ∀ (total_bananas : ℕ),
  (∃ (initial_children : ℕ), total_bananas = 2 * initial_children) →
  (∃ (present_children : ℕ), present_children = initial_children - 390) →
  (total_bananas = 4 * present_children) →
  initial_children = 780 := by
sorry

end NUMINAMATH_CALUDE_school_children_count_l3777_377784


namespace NUMINAMATH_CALUDE_percentage_calculation_l3777_377754

theorem percentage_calculation (P : ℝ) : 
  (0.47 * 1442 - P / 100 * 1412) + 65 = 5 → P = 52.24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3777_377754


namespace NUMINAMATH_CALUDE_average_weight_l3777_377749

theorem average_weight (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 70)
  (avg_bc : (b + c) / 2 = 50)
  (weight_b : b = 60) :
  (a + b + c) / 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_l3777_377749


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3777_377721

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * x - 1
  f (-1/3) = 0 ∧ f 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3777_377721


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3777_377701

theorem quadratic_roots_problem (p q : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (p + 5 * Complex.I) ^ 2 - (12 + 8 * Complex.I) * (p + 5 * Complex.I) + (20 + 40 * Complex.I) = 0 →
  (q + 3 * Complex.I) ^ 2 - (12 + 8 * Complex.I) * (q + 3 * Complex.I) + (20 + 40 * Complex.I) = 0 →
  p = 10 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3777_377701


namespace NUMINAMATH_CALUDE_pencil_distribution_l3777_377757

theorem pencil_distribution (initial_pencils : ℕ) (kept_pencils : ℕ) (extra_to_nilo : ℕ) 
  (h1 : initial_pencils = 50)
  (h2 : kept_pencils = 20)
  (h3 : extra_to_nilo = 10) : 
  ∃ (pencils_to_manny : ℕ), 
    pencils_to_manny + (pencils_to_manny + extra_to_nilo) = initial_pencils - kept_pencils ∧ 
    pencils_to_manny = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3777_377757


namespace NUMINAMATH_CALUDE_parallelogram_with_equal_vector_sums_is_rectangle_l3777_377764

/-- A parallelogram ABCD with vertices A, B, C, and D. -/
structure Parallelogram (V : Type*) [NormedAddCommGroup V] :=
  (A B C D : V)
  (is_parallelogram : (B - A) = (C - D) ∧ (D - A) = (C - B))

/-- Definition of a rectangle as a parallelogram with equal diagonals. -/
def is_rectangle {V : Type*} [NormedAddCommGroup V] (p : Parallelogram V) : Prop :=
  ‖p.C - p.A‖ = ‖p.D - p.B‖

theorem parallelogram_with_equal_vector_sums_is_rectangle
  {V : Type*} [NormedAddCommGroup V] (p : Parallelogram V) :
  ‖p.B - p.A + (p.D - p.A)‖ = ‖p.B - p.A - (p.D - p.A)‖ →
  is_rectangle p :=
sorry

end NUMINAMATH_CALUDE_parallelogram_with_equal_vector_sums_is_rectangle_l3777_377764


namespace NUMINAMATH_CALUDE_number_selection_game_probability_l3777_377746

/-- The probability of not winning a prize in the number selection game -/
def prob_not_win : ℚ := 2499 / 2500

/-- The number of options to choose from -/
def num_options : ℕ := 50

theorem number_selection_game_probability :
  prob_not_win = 1 - (1 / num_options^2) :=
sorry

end NUMINAMATH_CALUDE_number_selection_game_probability_l3777_377746


namespace NUMINAMATH_CALUDE_white_balls_count_l3777_377740

theorem white_balls_count (red_balls : ℕ) (total_balls : ℕ) (white_balls : ℕ) : 
  red_balls = 3 →
  (red_balls : ℚ) / total_balls = 1 / 4 →
  total_balls = red_balls + white_balls →
  white_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l3777_377740


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l3777_377769

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : Point)

-- Define properties of a quadrilateral
def has_one_pair_parallel_sides (q : Quadrilateral) : Prop := sorry
def has_one_pair_equal_sides (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- The main theorem
theorem quadrilateral_properties :
  (∃ q : Quadrilateral, has_one_pair_parallel_sides q ∧ has_one_pair_equal_sides q ∧ ¬is_parallelogram q) ∧
  (∀ q : Quadrilateral, is_parallelogram q → has_one_pair_parallel_sides q ∧ has_one_pair_equal_sides q) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l3777_377769


namespace NUMINAMATH_CALUDE_expression_evaluation_l3777_377795

theorem expression_evaluation (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5*y) * (-x - 5*y) - (-x + 5*y)^2 = -5.5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3777_377795


namespace NUMINAMATH_CALUDE_C14_not_allotrope_C60_l3777_377741

/-- Represents an atom -/
structure Atom where
  name : String

/-- Represents a molecule -/
structure Molecule where
  name : String

/-- Defines the concept of allotrope -/
def is_allotrope (a b : Atom) : Prop :=
  ∃ (element : String), a.name = element ∧ b.name = element

/-- C14 is an atom -/
def C14 : Atom := ⟨"C14"⟩

/-- C60 is a molecule -/
def C60 : Molecule := ⟨"C60"⟩

/-- Theorem stating that C14 is not an allotrope of C60 -/
theorem C14_not_allotrope_C60 : ¬∃ (a : Atom), is_allotrope C14 a ∧ a.name = C60.name := by
  sorry

end NUMINAMATH_CALUDE_C14_not_allotrope_C60_l3777_377741


namespace NUMINAMATH_CALUDE_ellipse_equation_l3777_377774

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis : ℝ
  focal_distance : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / (e.major_axis/2)^2 + y^2 / ((e.major_axis/2)^2 - (e.focal_distance/2)^2) = 1

/-- Theorem stating the standard equation for a specific ellipse -/
theorem ellipse_equation (e : Ellipse) 
    (h1 : e.center = (0, 0))
    (h2 : e.major_axis = 18)
    (h3 : e.focal_distance = 12) :
    ∀ x y : ℝ, standard_equation e x y ↔ x^2/81 + y^2/45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3777_377774


namespace NUMINAMATH_CALUDE_fraction_inequality_l3777_377783

theorem fraction_inequality (x : ℝ) : 
  -3 ≤ x ∧ x ≤ 1 ∧ (3 * x + 8 ≥ 3 * (5 - 2 * x)) → 7/9 ≤ x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3777_377783


namespace NUMINAMATH_CALUDE_average_age_calculation_l3777_377788

theorem average_age_calculation (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 36 →
  let total_age := num_students * avg_age_students + num_parents * avg_age_parents
  let total_people := num_students + num_parents
  (total_age / total_people : ℚ) = 26.4 := by
sorry

end NUMINAMATH_CALUDE_average_age_calculation_l3777_377788


namespace NUMINAMATH_CALUDE_coconuts_yield_five_l3777_377742

/-- The number of coconuts each tree yields -/
def coconuts_per_tree (price_per_coconut : ℚ) (total_amount : ℚ) (num_trees : ℕ) : ℚ :=
  (total_amount / price_per_coconut) / num_trees

/-- Proof that each tree yields 5 coconuts given the conditions -/
theorem coconuts_yield_five :
  coconuts_per_tree 3 90 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_coconuts_yield_five_l3777_377742


namespace NUMINAMATH_CALUDE_square_sum_equality_l3777_377735

theorem square_sum_equality : 12^2 + 2*(12*5) + 5^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3777_377735


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l3777_377773

/-- A quadratic function f(x) = 4x² - mx + 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f m x < f m y

/-- If f(x) = 4x² - mx + 5 is increasing on [-2, +∞), then m ≤ -16 -/
theorem quadratic_increasing_condition (m : ℝ) :
  is_increasing_on_interval m → m ≤ -16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l3777_377773


namespace NUMINAMATH_CALUDE_monroe_made_200_granola_bars_l3777_377772

/-- The number of granola bars Monroe made -/
def total_granola_bars : ℕ := sorry

/-- The number of granola bars eaten by Monroe and her husband -/
def eaten_by_parents : ℕ := 80

/-- The number of children in Monroe's family -/
def number_of_children : ℕ := 6

/-- The number of granola bars each child received -/
def bars_per_child : ℕ := 20

/-- Theorem stating that Monroe made 200 granola bars -/
theorem monroe_made_200_granola_bars :
  total_granola_bars = eaten_by_parents + number_of_children * bars_per_child :=
sorry

end NUMINAMATH_CALUDE_monroe_made_200_granola_bars_l3777_377772


namespace NUMINAMATH_CALUDE_oil_redistribution_l3777_377725

theorem oil_redistribution (trucks_a : Nat) (boxes_a : Nat) (trucks_b : Nat) (boxes_b : Nat) 
  (containers_per_box : Nat) (new_trucks : Nat) :
  trucks_a = 7 →
  boxes_a = 20 →
  trucks_b = 5 →
  boxes_b = 12 →
  containers_per_box = 8 →
  new_trucks = 10 →
  (trucks_a * boxes_a + trucks_b * boxes_b) * containers_per_box / new_trucks = 160 := by
  sorry

end NUMINAMATH_CALUDE_oil_redistribution_l3777_377725


namespace NUMINAMATH_CALUDE_change_ways_50_cents_l3777_377737

/-- Represents the number of ways to make change for a given amount using pennies, nickels, and dimes. -/
def changeWays (amount : ℕ) : ℕ := sorry

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a dime in cents -/
def dimeValue : ℕ := 10

/-- The total amount we want to make change for, in cents -/
def totalAmount : ℕ := 50

theorem change_ways_50_cents :
  changeWays totalAmount = 35 := by sorry

end NUMINAMATH_CALUDE_change_ways_50_cents_l3777_377737


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l3777_377799

theorem quadratic_root_k_value (k : ℝ) : 
  (2 : ℝ)^2 - k = 5 → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l3777_377799


namespace NUMINAMATH_CALUDE_second_number_is_72_l3777_377747

theorem second_number_is_72 (a b c : ℚ) : 
  a + b + c = 264 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a → 
  b = 72 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_72_l3777_377747


namespace NUMINAMATH_CALUDE_class_average_score_l3777_377731

theorem class_average_score (total_students : ℕ) 
  (score_95_count score_0_count score_65_count score_80_count : ℕ)
  (remaining_avg : ℚ) :
  total_students = 40 →
  score_95_count = 5 →
  score_0_count = 3 →
  score_65_count = 6 →
  score_80_count = 8 →
  remaining_avg = 45 →
  (2000 : ℚ) ≤ (score_95_count * 95 + score_0_count * 0 + score_65_count * 65 + 
    score_80_count * 80 + (total_students - score_95_count - score_0_count - 
    score_65_count - score_80_count) * remaining_avg) →
  (score_95_count * 95 + score_0_count * 0 + score_65_count * 65 + 
    score_80_count * 80 + (total_students - score_95_count - score_0_count - 
    score_65_count - score_80_count) * remaining_avg) ≤ (2400 : ℚ) →
  (score_95_count * 95 + score_0_count * 0 + score_65_count * 65 + 
    score_80_count * 80 + (total_students - score_95_count - score_0_count - 
    score_65_count - score_80_count) * remaining_avg) / total_students = (57875 : ℚ) / 1000 :=
by sorry

end NUMINAMATH_CALUDE_class_average_score_l3777_377731


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3777_377778

theorem triangle_angle_measure (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sine : (7 / Real.sin A) = (8 / Real.sin B) ∧ (8 / Real.sin B) = (13 / Real.sin C)) : 
  C = (2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3777_377778


namespace NUMINAMATH_CALUDE_largest_value_is_2_pow_35_l3777_377713

theorem largest_value_is_2_pow_35 : 
  (2 ^ 35 : ℕ) > 26 ∧ (2 ^ 35 : ℕ) > 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_is_2_pow_35_l3777_377713


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3777_377765

/-- A circle tangent to the y-axis and a line, passing through a specific point -/
structure TangentCircle where
  -- The slope of the line the circle is tangent to
  slope : ℝ
  -- The point the circle passes through
  point : ℝ × ℝ

/-- The radii of a circle satisfying the given conditions -/
def circle_radii (c : TangentCircle) : Set ℝ :=
  {r : ℝ | r = 1 ∨ r = 7/3}

/-- Theorem stating that a circle satisfying the given conditions has radius 1 or 7/3 -/
theorem tangent_circle_radius 
  (c : TangentCircle) 
  (h1 : c.slope = Real.sqrt 3 / 3) 
  (h2 : c.point = (2, Real.sqrt 3)) : 
  ∀ r ∈ circle_radii c, r = 1 ∨ r = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3777_377765


namespace NUMINAMATH_CALUDE_new_shoes_cost_proof_l3777_377751

/-- The cost of repairing used shoes -/
def repair_cost : ℝ := 10.50

/-- The duration (in years) that repaired used shoes last -/
def repair_duration : ℝ := 1

/-- The duration (in years) that new shoes last -/
def new_duration : ℝ := 2

/-- The percentage increase in average cost per year of new shoes compared to repaired shoes -/
def cost_increase_percentage : ℝ := 42.857142857142854

/-- The cost of purchasing new shoes -/
def new_shoes_cost : ℝ := 30.00

/-- Theorem stating that the cost of new shoes is $30.00 given the problem conditions -/
theorem new_shoes_cost_proof :
  new_shoes_cost = (repair_cost / repair_duration + cost_increase_percentage / 100 * repair_cost) * new_duration :=
by sorry

end NUMINAMATH_CALUDE_new_shoes_cost_proof_l3777_377751


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l3777_377771

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 104 → ¬(∃ k : ℕ, n = 13 * k) :=
by
  sorry

#check smallest_three_digit_multiple_of_13

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l3777_377771


namespace NUMINAMATH_CALUDE_salt_solution_volume_salt_solution_volume_proof_l3777_377759

/-- Given a solution with initial salt concentration of 10% that becomes 8% salt
    after adding 16 gallons of water, prove the initial volume is 64 gallons. -/
theorem salt_solution_volume : ℝ → Prop :=
  fun initial_volume =>
    let initial_salt_concentration : ℝ := 0.10
    let final_salt_concentration : ℝ := 0.08
    let added_water : ℝ := 16
    let final_volume : ℝ := initial_volume + added_water
    initial_salt_concentration * initial_volume =
      final_salt_concentration * final_volume →
    initial_volume = 64

/-- Proof of the salt_solution_volume theorem -/
theorem salt_solution_volume_proof : salt_solution_volume 64 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_volume_salt_solution_volume_proof_l3777_377759


namespace NUMINAMATH_CALUDE_function_properties_l3777_377704

-- Define the function f(x) = ax³ + bx - 1
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 1

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem function_properties :
  ∃ (a b : ℝ),
    (f a b 1 = -3) ∧
    (f_derivative a b 1 = 0) ∧
    (a = 1) ∧
    (b = -3) ∧
    (∀ x ∈ Set.Icc (-2) 3, f a b x ≤ 17) ∧
    (∃ x ∈ Set.Icc (-2) 3, f a b x = 17) ∧
    (∀ x ∈ Set.Icc (-2) 3, f a b x ≥ -3) ∧
    (∃ x ∈ Set.Icc (-2) 3, f a b x = -3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3777_377704


namespace NUMINAMATH_CALUDE_range_of_m_inequality_for_nonzero_x_l3777_377724

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m|

-- Theorem 1: Range of m
theorem range_of_m (m : ℝ) : 
  (f m 1 + f m (-2) ≥ 5) ↔ (m ≤ -2 ∨ m ≥ 3) := by sorry

-- Theorem 2: Inequality for non-zero x
theorem inequality_for_nonzero_x (m : ℝ) (x : ℝ) (h : x ≠ 0) : 
  f m (1/x) + f m (-x) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_inequality_for_nonzero_x_l3777_377724


namespace NUMINAMATH_CALUDE_special_sequence_a9_l3777_377785

/-- A sequence of positive integers satisfying the given property -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ p q : ℕ, a (p + q) = a p + a q)

theorem special_sequence_a9 (a : ℕ → ℕ) (h : SpecialSequence a) (h2 : a 2 = 4) : 
  a 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_a9_l3777_377785


namespace NUMINAMATH_CALUDE_thousand_to_hundred_power_l3777_377727

theorem thousand_to_hundred_power (h : 1000 = 10^3) : 1000^100 = 10^300 := by
  sorry

end NUMINAMATH_CALUDE_thousand_to_hundred_power_l3777_377727


namespace NUMINAMATH_CALUDE_spells_conversion_l3777_377703

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The number of spells in each book in base-9 --/
def spellsPerBook : List Nat := [5, 3, 6]

theorem spells_conversion :
  base9ToBase10 spellsPerBook = 518 := by
  sorry

#eval base9ToBase10 spellsPerBook

end NUMINAMATH_CALUDE_spells_conversion_l3777_377703


namespace NUMINAMATH_CALUDE_fraction_addition_l3777_377728

theorem fraction_addition : (7 : ℚ) / 12 + (3 : ℚ) / 8 = (23 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3777_377728


namespace NUMINAMATH_CALUDE_rice_grains_difference_l3777_377716

def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_of_grains (n : ℕ) : ℕ := 
  3 * (3^n - 1) / 2

theorem rice_grains_difference : 
  grains_on_square 11 - sum_of_grains 9 = 147624 := by
  sorry

end NUMINAMATH_CALUDE_rice_grains_difference_l3777_377716


namespace NUMINAMATH_CALUDE_shaded_percentage_of_grid_l3777_377729

theorem shaded_percentage_of_grid (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 25 →
  shaded_squares = 13 →
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_shaded_percentage_of_grid_l3777_377729


namespace NUMINAMATH_CALUDE_train_passing_jogger_l3777_377723

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (initial_distance : ℝ) 
  (train_length : ℝ) 
  (h1 : jogger_speed = 9 * (1000 / 3600))  -- 9 kmph in m/s
  (h2 : train_speed = 45 * (1000 / 3600))  -- 45 kmph in m/s
  (h3 : initial_distance = 240)
  (h4 : train_length = 120) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) = 36 := by
  sorry


end NUMINAMATH_CALUDE_train_passing_jogger_l3777_377723


namespace NUMINAMATH_CALUDE_base_subtraction_l3777_377767

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem base_subtraction :
  let base9_321 := [3, 2, 1]
  let base6_254 := [2, 5, 4]
  (toBase10 base9_321 9) - (toBase10 base6_254 6) = 156 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l3777_377767


namespace NUMINAMATH_CALUDE_cost_price_change_l3777_377793

theorem cost_price_change (x : ℝ) : 
  (50 * (1 + x / 100) * (1 - x / 100) = 48) → x = 20 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_change_l3777_377793


namespace NUMINAMATH_CALUDE_chef_inventory_solution_l3777_377719

def chef_inventory (initial_apples initial_flour initial_sugar initial_butter : ℕ) : Prop :=
  let used_apples : ℕ := 15
  let used_flour : ℕ := 6
  let used_sugar : ℕ := 14  -- 10 initially + 4 from newly bought
  let used_butter : ℕ := 3
  let remaining_apples : ℕ := 4
  let remaining_flour : ℕ := 3
  let remaining_sugar : ℕ := 13
  let remaining_butter : ℕ := 2
  let given_away_apples : ℕ := 2
  (initial_apples = used_apples + given_away_apples + remaining_apples) ∧
  (initial_flour = 2 * (used_flour + remaining_flour)) ∧
  (initial_sugar = used_sugar + remaining_sugar) ∧
  (initial_butter = used_butter + remaining_butter)

theorem chef_inventory_solution :
  ∃ (initial_apples initial_flour initial_sugar initial_butter : ℕ),
    chef_inventory initial_apples initial_flour initial_sugar initial_butter ∧
    initial_apples = 21 ∧
    initial_flour = 18 ∧
    initial_sugar = 27 ∧
    initial_butter = 5 ∧
    initial_apples + initial_flour + initial_sugar + initial_butter = 71 :=
by sorry

end NUMINAMATH_CALUDE_chef_inventory_solution_l3777_377719


namespace NUMINAMATH_CALUDE_frannie_jump_count_l3777_377722

/-- The number of times Meg jumped -/
def meg_jumps : ℕ := 71

/-- The difference between Meg's and Frannie's jumps -/
def jump_difference : ℕ := 18

/-- The number of times Frannie jumped -/
def frannie_jumps : ℕ := meg_jumps - jump_difference

theorem frannie_jump_count : frannie_jumps = 53 := by sorry

end NUMINAMATH_CALUDE_frannie_jump_count_l3777_377722


namespace NUMINAMATH_CALUDE_symmetric_sine_cosine_value_l3777_377796

/-- Given a function f(x) = 3sin(ωx + φ) that is symmetric about x = π/3,
    prove that g(π/3) = 1 where g(x) = 3cos(ωx + φ) + 1 -/
theorem symmetric_sine_cosine_value 
  (ω φ : ℝ) 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (hf : f = fun x ↦ 3 * Real.sin (ω * x + φ))
  (hg : g = fun x ↦ 3 * Real.cos (ω * x + φ) + 1)
  (h_sym : ∀ x : ℝ, f (π / 3 + x) = f (π / 3 - x)) : 
  g (π / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sine_cosine_value_l3777_377796


namespace NUMINAMATH_CALUDE_pentagon_smallest_angle_l3777_377707

theorem pentagon_smallest_angle 
  (angles : Fin 5 → ℝ)
  (arithmetic_sequence : ∀ i : Fin 4, angles (i + 1) - angles i = angles (i + 2) - angles (i + 1))
  (largest_angle : angles 4 = 150)
  (angle_sum : angles 0 + angles 1 + angles 2 + angles 3 + angles 4 = 540) :
  angles 0 = 66 := by
sorry

end NUMINAMATH_CALUDE_pentagon_smallest_angle_l3777_377707


namespace NUMINAMATH_CALUDE_range_of_a_l3777_377712

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 - x + a

-- Define the range of f as set B
def B (a : ℝ) : Set ℝ := {y : ℝ | ∃ x ∈ A, f a x = y}

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ A, f a x ∈ A) → a = -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3777_377712


namespace NUMINAMATH_CALUDE_cost_of_cakes_l3777_377714

/-- The cost of cakes problem -/
theorem cost_of_cakes (num_cakes : ℕ) (johns_share : ℚ) (cost_per_cake : ℚ) 
  (h1 : num_cakes = 3)
  (h2 : johns_share = 18)
  (h3 : johns_share * 2 = num_cakes * cost_per_cake) :
  cost_per_cake = 12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_cakes_l3777_377714


namespace NUMINAMATH_CALUDE_expression_value_l3777_377787

theorem expression_value : 
  Real.sqrt 3 / 2 - Real.sqrt 3 * (Real.sin (15 * π / 180))^2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3777_377787


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l3777_377732

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = !![3, 4; -2, -3] →
  (B^3)⁻¹ = !![3, 4; -2, -3] :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l3777_377732


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_49_l3777_377798

theorem factor_t_squared_minus_49 : ∀ t : ℝ, t^2 - 49 = (t - 7) * (t + 7) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_49_l3777_377798


namespace NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l3777_377738

theorem consecutive_integers_fourth_power_sum (a b c : ℤ) : 
  (b = a + 1) →
  (c = b + 1) →
  (a^2 + b^2 + c^2 = 12246) →
  (a^4 + b^4 + c^4 = 50380802) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l3777_377738


namespace NUMINAMATH_CALUDE_octagon_area_equal_perimeter_l3777_377792

theorem octagon_area_equal_perimeter (s : Real) (o : Real) : 
  s > 0 → o > 0 →
  s^2 = 16 →
  4 * s = 8 * o →
  2 * (1 + Real.sqrt 2) * o^2 = 8 + 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_equal_perimeter_l3777_377792


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3777_377779

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + k = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 - 3*y + k = 0 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3777_377779
