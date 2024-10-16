import Mathlib

namespace NUMINAMATH_CALUDE_f_def_f_5_eq_0_l1628_162896

def f (x : ℝ) : ℝ := sorry

theorem f_def (x : ℝ) : f (2 * x + 1) = x^2 - 2*x := sorry

theorem f_5_eq_0 : f 5 = 0 := by sorry

end NUMINAMATH_CALUDE_f_def_f_5_eq_0_l1628_162896


namespace NUMINAMATH_CALUDE_geometric_progression_seventh_term_l1628_162846

-- Define the geometric progression
def geometric_progression (b₁ q : ℚ) (n : ℕ) : ℚ :=
  b₁ * q^(n-1)

-- Define the conditions
def condition1 (b₁ q : ℚ) : Prop :=
  b₁ + b₁*q + b₁*q^2 = 91

def condition2 (b₁ q : ℚ) : Prop :=
  (b₁*q + 27) - (b₁ + 25) = (b₁*q^2 + 1) - (b₁*q + 27)

-- Theorem statement
theorem geometric_progression_seventh_term
  (b₁ q : ℚ) (h1 : condition1 b₁ q) (h2 : condition2 b₁ q) :
  (geometric_progression b₁ q 7 = (35 * 46656) / 117649) ∨
  (geometric_progression b₁ q 7 = (63 * 4096) / 117649) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_seventh_term_l1628_162846


namespace NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l1628_162808

theorem odd_divides_power_factorial_minus_one (n : ℕ) (h : Odd n) : n ∣ 2^(n.factorial) - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l1628_162808


namespace NUMINAMATH_CALUDE_plate_difference_l1628_162871

/- Define the number of kitchen supplies for Angela and Sharon -/
def angela_pots : ℕ := 20
def angela_plates : ℕ := 3 * angela_pots + 6
def angela_cutlery : ℕ := angela_plates / 2

def sharon_pots : ℕ := angela_pots / 2
def sharon_cutlery : ℕ := angela_cutlery * 2
def sharon_total : ℕ := 254

/- Define Sharon's plates as the remaining items after subtracting pots and cutlery from the total -/
def sharon_plates : ℕ := sharon_total - (sharon_pots + sharon_cutlery)

/- Theorem stating the difference between Sharon's plates and three times Angela's plates -/
theorem plate_difference : 
  3 * angela_plates - sharon_plates = 20 := by sorry

end NUMINAMATH_CALUDE_plate_difference_l1628_162871


namespace NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l1628_162886

def S : Finset ℕ := Finset.range 9

def difference_sum (S : Finset ℕ) : ℕ :=
  S.sum (λ i => S.sum (λ j => if i > j then (3^i - 3^j) else 0))

theorem difference_sum_of_powers_of_three : difference_sum S = 68896 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l1628_162886


namespace NUMINAMATH_CALUDE_investment_comparison_l1628_162899

def initial_investment : ℝ := 200

def delta_year1_change : ℝ := 1.10
def delta_year2_change : ℝ := 0.90

def echo_year1_change : ℝ := 0.70
def echo_year2_change : ℝ := 1.50

def foxtrot_year1_change : ℝ := 1.00
def foxtrot_year2_change : ℝ := 0.95

def final_delta : ℝ := initial_investment * delta_year1_change * delta_year2_change
def final_echo : ℝ := initial_investment * echo_year1_change * echo_year2_change
def final_foxtrot : ℝ := initial_investment * foxtrot_year1_change * foxtrot_year2_change

theorem investment_comparison : final_foxtrot < final_delta ∧ final_delta < final_echo := by
  sorry

end NUMINAMATH_CALUDE_investment_comparison_l1628_162899


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l1628_162872

theorem matching_shoes_probability (total_shoes : ℕ) (total_pairs : ℕ) (h1 : total_shoes = 12) (h2 : total_pairs = 6) :
  let total_selections := total_shoes.choose 2
  let matching_selections := total_pairs
  (matching_selections : ℚ) / total_selections = 1 / 11 := by sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l1628_162872


namespace NUMINAMATH_CALUDE_sports_love_distribution_l1628_162845

theorem sports_love_distribution (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (boys_not_love_sports : ℕ) (total_not_love_sports : ℕ) :
  total_students = 50 →
  total_boys = 30 →
  total_girls = 20 →
  boys_not_love_sports = 12 →
  total_not_love_sports = 24 →
  ∃ (boys_love_sports : ℕ) (total_love_sports : ℕ),
    boys_love_sports = total_boys - boys_not_love_sports ∧
    total_love_sports = total_students - total_not_love_sports ∧
    boys_love_sports = 18 ∧
    total_love_sports = 26 := by
  sorry

end NUMINAMATH_CALUDE_sports_love_distribution_l1628_162845


namespace NUMINAMATH_CALUDE_prime_power_equation_solutions_l1628_162800

theorem prime_power_equation_solutions :
  ∀ p n : ℕ,
    Nat.Prime p →
    n > 0 →
    p^3 - 2*p^2 + p + 1 = 3^n →
    ((p = 2 ∧ n = 1) ∨ (p = 5 ∧ n = 4)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_equation_solutions_l1628_162800


namespace NUMINAMATH_CALUDE_plane_speed_calculation_l1628_162838

theorem plane_speed_calculation (D : ℝ) (V : ℝ) (h1 : D = V * 5) (h2 : D = 720 * (5/3)) :
  V = 240 := by
sorry

end NUMINAMATH_CALUDE_plane_speed_calculation_l1628_162838


namespace NUMINAMATH_CALUDE_borgnine_leg_count_l1628_162861

/-- The number of legs Borgnine wants to see at the zoo -/
def total_legs (chimps lions lizards tarantulas : ℕ) : ℕ :=
  2 * chimps + 4 * lions + 4 * lizards + 8 * tarantulas

/-- Theorem stating the total number of legs Borgnine wants to see -/
theorem borgnine_leg_count : total_legs 12 8 5 125 = 1076 := by
  sorry

end NUMINAMATH_CALUDE_borgnine_leg_count_l1628_162861


namespace NUMINAMATH_CALUDE_R_symmetry_l1628_162805

/-- Recursive definition of R_n sequences -/
def R : ℕ → List ℕ
  | 0 => [1]
  | n + 1 =>
    let prev := R n
    List.join (prev.map (fun x => List.range x)) ++ [n + 1]

/-- Main theorem -/
theorem R_symmetry (n : ℕ) (k : ℕ) (h : n > 1) :
  (R n).nthLe k (by sorry) = 1 ↔
  (R n).nthLe ((R n).length - 1 - k) (by sorry) ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_R_symmetry_l1628_162805


namespace NUMINAMATH_CALUDE_square_circumference_l1628_162874

/-- Given a square with an area of 324 square meters, its circumference is 72 meters. -/
theorem square_circumference (s : Real) (area : Real) (h1 : area = 324) (h2 : s^2 = area) :
  4 * s = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_circumference_l1628_162874


namespace NUMINAMATH_CALUDE_kelly_initial_games_l1628_162828

/-- The number of Nintendo games Kelly gave away -/
def games_given_away : ℕ := 64

/-- The number of Nintendo games Kelly has left -/
def games_left : ℕ := 42

/-- The initial number of Nintendo games Kelly had -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_initial_games : initial_games = 106 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_games_l1628_162828


namespace NUMINAMATH_CALUDE_highest_probability_prime_l1628_162812

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_factor_of_12 (n : ℕ) : Prop := 12 % n = 0

def total_outcomes : ℕ := 36

def prime_outcomes : ℕ := 15
def multiple_of_4_outcomes : ℕ := 9
def perfect_square_outcomes : ℕ := 7
def score_7_outcomes : ℕ := 6
def factor_of_12_outcomes : ℕ := 12

theorem highest_probability_prime :
  prime_outcomes > multiple_of_4_outcomes ∧
  prime_outcomes > perfect_square_outcomes ∧
  prime_outcomes > score_7_outcomes ∧
  prime_outcomes > factor_of_12_outcomes :=
sorry

end NUMINAMATH_CALUDE_highest_probability_prime_l1628_162812


namespace NUMINAMATH_CALUDE_current_speed_l1628_162815

/-- The speed of the current in a river, given the rowing speed in still water and the time taken to cover a certain distance downstream. -/
theorem current_speed (still_water_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  still_water_speed = 22 →
  downstream_distance = 80 →
  downstream_time = 11.519078473722104 →
  ∃ current_speed : ℝ, 
    (current_speed * 1000 / 3600 + still_water_speed * 1000 / 3600) * downstream_time = downstream_distance ∧ 
    abs (current_speed - 2.9988) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l1628_162815


namespace NUMINAMATH_CALUDE_golden_ratio_unique_progression_l1628_162820

theorem golden_ratio_unique_progression : ∃! x : ℝ, 
  x > 0 ∧ 
  let b := ⌊x⌋
  let c := x - b
  (c < b) ∧ (b < x) ∧ (c * x = b * b) ∧ x = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_unique_progression_l1628_162820


namespace NUMINAMATH_CALUDE_joe_hvac_zones_l1628_162850

def hvac_system (total_cost : ℕ) (vents_per_zone : ℕ) (cost_per_vent : ℕ) : ℕ :=
  (total_cost / cost_per_vent) / vents_per_zone

theorem joe_hvac_zones :
  hvac_system 20000 5 2000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joe_hvac_zones_l1628_162850


namespace NUMINAMATH_CALUDE_heesu_has_greatest_sum_l1628_162841

-- Define the card values for each person
def sora_cards : List Nat := [4, 6]
def heesu_cards : List Nat := [7, 5]
def jiyeon_cards : List Nat := [3, 8]

-- Define a function to calculate the sum of cards
def sum_cards (cards : List Nat) : Nat :=
  cards.sum

-- Theorem statement
theorem heesu_has_greatest_sum :
  sum_cards heesu_cards > sum_cards sora_cards ∧
  sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  sorry


end NUMINAMATH_CALUDE_heesu_has_greatest_sum_l1628_162841


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_perimeter_difference_unique_d_value_count_impossible_d_values_l1628_162807

theorem isosceles_triangle_rectangle_perimeter_difference 
  (d : ℕ) (w : ℝ) : 
  w > 0 → 
  6 * w > 0 → 
  6 * w + 2 * d = 6 * w + 1236 → 
  d = 618 := by
sorry

theorem unique_d_value : 
  ∃! d : ℕ, ∃ w : ℝ, 
    w > 0 ∧ 
    6 * w > 0 ∧ 
    6 * w + 2 * d = 6 * w + 1236 := by
sorry

theorem count_impossible_d_values : 
  (Nat.card {d : ℕ | ¬∃ w : ℝ, w > 0 ∧ 6 * w > 0 ∧ 6 * w + 2 * d = 6 * w + 1236}) = ℵ₀ := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_perimeter_difference_unique_d_value_count_impossible_d_values_l1628_162807


namespace NUMINAMATH_CALUDE_apps_deleted_l1628_162840

theorem apps_deleted (initial_apps new_apps remaining_apps : ℕ) : 
  initial_apps = 10 →
  new_apps = 11 →
  remaining_apps = 4 →
  initial_apps + new_apps - remaining_apps = 17 :=
by sorry

end NUMINAMATH_CALUDE_apps_deleted_l1628_162840


namespace NUMINAMATH_CALUDE_art_class_selection_l1628_162847

theorem art_class_selection (n m k : ℕ) (h1 : n = 10) (h2 : m = 4) (h3 : k = 2) :
  (Nat.choose (n - k + 1) (m - k)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_art_class_selection_l1628_162847


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1628_162844

/-- A function to check if three line segments can form a right triangle -/
def isRightTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Theorem stating that among the given sets, only {2, √2, √2} forms a right triangle -/
theorem right_triangle_sets :
  ¬ isRightTriangle 2 3 4 ∧
  ¬ isRightTriangle 1 1 2 ∧
  ¬ isRightTriangle 4 5 6 ∧
  isRightTriangle 2 (Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1628_162844


namespace NUMINAMATH_CALUDE_monthly_subscription_more_cost_effective_l1628_162804

/-- Represents the cost of internet access plans -/
def internet_cost (pay_per_minute_rate : ℚ) (monthly_fee : ℚ) (communication_fee : ℚ) (hours : ℚ) : ℚ × ℚ :=
  let minutes : ℚ := hours * 60
  let pay_per_minute_cost : ℚ := (pay_per_minute_rate + communication_fee) * minutes
  let monthly_subscription_cost : ℚ := monthly_fee + communication_fee * minutes
  (pay_per_minute_cost, monthly_subscription_cost)

theorem monthly_subscription_more_cost_effective :
  let (pay_per_minute_cost, monthly_subscription_cost) :=
    internet_cost (5 / 100) 50 (2 / 100) 20
  monthly_subscription_cost < pay_per_minute_cost :=
by sorry

end NUMINAMATH_CALUDE_monthly_subscription_more_cost_effective_l1628_162804


namespace NUMINAMATH_CALUDE_negative_result_operations_l1628_162891

theorem negative_result_operations : 
  (-(-4) > 0) ∧ 
  (abs (-4) > 0) ∧ 
  (-4^2 < 0) ∧ 
  ((-4)^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_result_operations_l1628_162891


namespace NUMINAMATH_CALUDE_closer_to_center_is_enclosed_by_bisectors_l1628_162839

/-- A rectangle in a 2D plane -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of points closer to the center of a rectangle than to any of its vertices -/
def CloserToCenter (r : Rectangle) : Set Point :=
  { p : Point | p.x^2 + p.y^2 < (p.x - r.a)^2 + (p.y - r.b)^2 ∧
                p.x^2 + p.y^2 < (p.x - r.a)^2 + (p.y + r.b)^2 ∧
                p.x^2 + p.y^2 < (p.x + r.a)^2 + (p.y + r.b)^2 ∧
                p.x^2 + p.y^2 < (p.x + r.a)^2 + (p.y - r.b)^2 }

/-- Theorem stating that the set of points closer to the center is enclosed by perpendicular bisectors -/
theorem closer_to_center_is_enclosed_by_bisectors (r : Rectangle) :
  ∃ (bisectors : Set Point), CloserToCenter r = bisectors :=
sorry

end NUMINAMATH_CALUDE_closer_to_center_is_enclosed_by_bisectors_l1628_162839


namespace NUMINAMATH_CALUDE_unit_segment_construction_l1628_162842

theorem unit_segment_construction (a : ℝ) (h : a > 1) : (a / a^2) * a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_segment_construction_l1628_162842


namespace NUMINAMATH_CALUDE_frog_jump_probability_l1628_162827

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The grid dimensions -/
def gridWidth : ℕ := 7
def gridHeight : ℕ := 5

/-- The jump distance -/
def jumpDistance : ℕ := 2

/-- The starting point of the frog -/
def startPoint : Point := ⟨2, 3⟩

/-- Predicate to check if a point is on a horizontal edge -/
def isOnHorizontalEdge (p : Point) : Prop :=
  p.y = 0 ∨ p.y = gridHeight

/-- Predicate to check if a point is on the grid -/
def isOnGrid (p : Point) : Prop :=
  p.x ≤ gridWidth ∧ p.y ≤ gridHeight

/-- The probability of reaching a horizontal edge from a given point -/
noncomputable def probReachHorizontalEdge (p : Point) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem frog_jump_probability :
  probReachHorizontalEdge startPoint = 3/4 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l1628_162827


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l1628_162819

theorem binomial_expansion_property (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + Real.sqrt 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_property_l1628_162819


namespace NUMINAMATH_CALUDE_annie_extracurricular_hours_l1628_162809

def chess_hours : ℕ := 2
def drama_hours : ℕ := 8
def glee_hours : ℕ := 3
def total_weeks : ℕ := 12
def sick_weeks : ℕ := 2

def extracurricular_hours_per_week : ℕ := chess_hours + drama_hours + glee_hours
def active_weeks : ℕ := total_weeks - sick_weeks

theorem annie_extracurricular_hours :
  extracurricular_hours_per_week * active_weeks = 130 := by
  sorry

end NUMINAMATH_CALUDE_annie_extracurricular_hours_l1628_162809


namespace NUMINAMATH_CALUDE_trapezoid_midline_length_l1628_162875

/-- Given a trapezoid with parallel sides of length a and b, 
    the length of the line segment joining the midpoints of these parallel sides is (a + b) / 2 -/
theorem trapezoid_midline_length (a b c d : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  let midline_length := (a + b) / 2
  midline_length = (a + b) / 2 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_midline_length_l1628_162875


namespace NUMINAMATH_CALUDE_inequality_proof_l1628_162884

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^2 + 1 - 2*x*(x*y^2 - x + z + 1) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1628_162884


namespace NUMINAMATH_CALUDE_ramsey_3_3_l1628_162816

/-- A complete graph with 6 vertices where each edge is colored either blue or red. -/
def ColoredGraph := Fin 6 → Fin 6 → Bool

/-- The graph is complete and each edge has a color (blue or red). -/
def is_valid_coloring (g : ColoredGraph) : Prop :=
  ∀ i j : Fin 6, i ≠ j → g i j = true ∨ g i j = false

/-- A triangle in the graph with all edges of the same color. -/
def monochromatic_triangle (g : ColoredGraph) : Prop :=
  ∃ i j k : Fin 6, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    ((g i j = true ∧ g j k = true ∧ g i k = true) ∨
     (g i j = false ∧ g j k = false ∧ g i k = false))

/-- The Ramsey theorem for R(3,3) -/
theorem ramsey_3_3 (g : ColoredGraph) (h : is_valid_coloring g) : 
  monochromatic_triangle g := by
  sorry

end NUMINAMATH_CALUDE_ramsey_3_3_l1628_162816


namespace NUMINAMATH_CALUDE_radii_of_circles_l1628_162806

/-- Two circles lying outside each other -/
structure TwoCircles where
  center_distance : ℝ
  external_tangent : ℝ
  internal_tangent : ℝ

/-- The radii of two circles -/
structure CircleRadii where
  r₁ : ℝ
  r₂ : ℝ

/-- Given the properties of two circles, compute their radii -/
def compute_radii (circles : TwoCircles) : CircleRadii :=
  { r₁ := 38, r₂ := 22 }

/-- Theorem stating that for the given circle properties, the radii are 38 and 22 -/
theorem radii_of_circles (circles : TwoCircles) 
    (h1 : circles.center_distance = 65) 
    (h2 : circles.external_tangent = 63) 
    (h3 : circles.internal_tangent = 25) : 
    compute_radii circles = { r₁ := 38, r₂ := 22 } := by
  sorry

end NUMINAMATH_CALUDE_radii_of_circles_l1628_162806


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l1628_162894

/-- The number of distinct ways to arrange n distinct beads on a bracelet, 
    considering rotations and reflections as equivalent -/
def bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements 
    for 8 beads is 2520 -/
theorem eight_bead_bracelet_arrangements : 
  bracelet_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l1628_162894


namespace NUMINAMATH_CALUDE_combined_contingency_funds_l1628_162892

/-- Calculates the combined contingency funds from two donations given specific conditions. -/
theorem combined_contingency_funds 
  (donation1 : ℝ) 
  (donation2 : ℝ) 
  (community_pantry_rate : ℝ) 
  (crisis_fund_rate : ℝ) 
  (livelihood_rate : ℝ) 
  (disaster_relief_rate : ℝ) 
  (international_aid_rate : ℝ) 
  (education_rate : ℝ) 
  (healthcare_rate : ℝ) 
  (conversion_rate : ℝ) :
  donation1 = 360 →
  donation2 = 180 →
  community_pantry_rate = 0.35 →
  crisis_fund_rate = 0.40 →
  livelihood_rate = 0.10 →
  disaster_relief_rate = 0.05 →
  international_aid_rate = 0.30 →
  education_rate = 0.25 →
  healthcare_rate = 0.25 →
  conversion_rate = 1.20 →
  (donation1 - (community_pantry_rate + crisis_fund_rate + livelihood_rate + disaster_relief_rate) * donation1) +
  (conversion_rate * donation2 - (international_aid_rate + education_rate + healthcare_rate) * conversion_rate * donation2) = 79.20 := by
  sorry


end NUMINAMATH_CALUDE_combined_contingency_funds_l1628_162892


namespace NUMINAMATH_CALUDE_duck_count_l1628_162855

theorem duck_count (total_legs : ℕ) (rabbit_count : ℕ) (rabbit_legs : ℕ) (duck_legs : ℕ) :
  total_legs = 48 →
  rabbit_count = 9 →
  rabbit_legs = 4 →
  duck_legs = 2 →
  (total_legs - rabbit_count * rabbit_legs) / duck_legs = 6 :=
by sorry

end NUMINAMATH_CALUDE_duck_count_l1628_162855


namespace NUMINAMATH_CALUDE_bethany_age_proof_l1628_162873

/-- Bethany's current age -/
def bethanys_current_age : ℕ := 19

/-- Bethany's younger sister's current age -/
def sisters_current_age : ℕ := 11

/-- Bethany's age three years ago -/
def bethanys_age_three_years_ago : ℕ := bethanys_current_age - 3

/-- Bethany's younger sister's age three years ago -/
def sisters_age_three_years_ago : ℕ := sisters_current_age - 3

theorem bethany_age_proof :
  (bethanys_age_three_years_ago = 2 * sisters_age_three_years_ago) ∧
  (sisters_current_age + 5 = 16) →
  bethanys_current_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_bethany_age_proof_l1628_162873


namespace NUMINAMATH_CALUDE_days_to_fill_tank_l1628_162856

def tank_capacity : ℝ := 350000 -- in milliliters
def min_daily_collection : ℝ := 1200 -- in milliliters
def max_daily_collection : ℝ := 2100 -- in milliliters

theorem days_to_fill_tank : 
  ∃ (days : ℕ), days = 213 ∧ 
  (tank_capacity / min_daily_collection ≤ days) ∧
  (tank_capacity / max_daily_collection ≤ days) ∧
  (∀ d : ℕ, d < days → d * max_daily_collection < tank_capacity) :=
sorry

end NUMINAMATH_CALUDE_days_to_fill_tank_l1628_162856


namespace NUMINAMATH_CALUDE_opposite_of_abs_neg_five_l1628_162868

theorem opposite_of_abs_neg_five : -(|-5|) = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_abs_neg_five_l1628_162868


namespace NUMINAMATH_CALUDE_bill_vote_difference_l1628_162814

theorem bill_vote_difference (total : ℕ) (initial_for initial_against revote_for revote_against : ℕ) :
  total = 400 →
  initial_for + initial_against = total →
  revote_for + revote_against = total →
  (revote_for : ℚ) - revote_against = 3 * (initial_against - initial_for) →
  (revote_for : ℚ) = 13 / 12 * initial_against →
  revote_for - initial_for = 36 :=
by sorry

end NUMINAMATH_CALUDE_bill_vote_difference_l1628_162814


namespace NUMINAMATH_CALUDE_birdhouse_revenue_theorem_l1628_162834

/-- Calculates the total revenue from selling birdhouses with discount and tax --/
def birdhouse_revenue (
  extra_large_price : ℚ)
  (large_price : ℚ)
  (medium_price : ℚ)
  (small_price : ℚ)
  (extra_small_price : ℚ)
  (extra_large_qty : ℕ)
  (large_qty : ℕ)
  (medium_qty : ℕ)
  (small_qty : ℕ)
  (extra_small_qty : ℕ)
  (discount_rate : ℚ)
  (tax_rate : ℚ) : ℚ :=
  let total_before_discount :=
    extra_large_price * extra_large_qty +
    large_price * large_qty +
    medium_price * medium_qty +
    small_price * small_qty +
    extra_small_price * extra_small_qty
  let discounted_amount := total_before_discount * (1 - discount_rate)
  let final_amount := discounted_amount * (1 + tax_rate)
  final_amount

/-- Theorem stating the total revenue from selling birdhouses --/
theorem birdhouse_revenue_theorem :
  birdhouse_revenue 45 22 16 10 5 3 5 7 8 10 (1/10) (6/100) = 464.60 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_revenue_theorem_l1628_162834


namespace NUMINAMATH_CALUDE_ellipse_properties_l1628_162824

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Focal distance of an ellipse -/
def focal_distance (e : Ellipse) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_properties (e : Ellipse) (l : Line) :
  l.point.1 = focal_distance e / 2 →  -- line passes through right focus
  l.slope = Real.tan (60 * π / 180) →  -- slope angle is 60°
  distance_point_to_line (-focal_distance e / 2, 0) l = 2 →  -- distance from left focus to line is 2
  focal_distance e = 4 ∧  -- focal distance is 4
  (e.b = 2 → e.a = 3 ∧ focal_distance e = 2 * Real.sqrt 5) :=  -- when b = 2, a = 3 and c = 2√5
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1628_162824


namespace NUMINAMATH_CALUDE_target_line_properties_l1628_162890

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def line2 (x y : ℝ) : Prop := x - y + 5 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y = 0
def target_line (x y : ℝ) : Prop := 2 * x + y - 8 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem target_line_properties :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    target_line x y ∧
    ∃ (m1 m2 : ℝ),
      (∀ (x y : ℝ), line3 x y ↔ y = m1 * x) ∧
      (∀ (x y : ℝ), target_line x y ↔ y = m2 * x + (y - m2 * x)) ∧
      perpendicular m1 m2 :=
sorry

end NUMINAMATH_CALUDE_target_line_properties_l1628_162890


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1628_162888

theorem determinant_of_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -1, 5]
  Matrix.det A = 44 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1628_162888


namespace NUMINAMATH_CALUDE_f_inequality_l1628_162883

/-- The number of ways to represent a positive integer as a sum of non-decreasing positive integers -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that f(n+1) is less than or equal to the average of f(n) and f(n+2) for any positive integer n -/
theorem f_inequality (n : ℕ) (h : n > 0) : f (n + 1) ≤ (f n + f (n + 2)) / 2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l1628_162883


namespace NUMINAMATH_CALUDE_flagpole_break_height_l1628_162858

theorem flagpole_break_height (h : ℝ) (b : ℝ) (x : ℝ) 
  (hypotenuse : h = 6)
  (base : b = 2)
  (right_triangle : x^2 + b^2 = h^2) :
  x = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l1628_162858


namespace NUMINAMATH_CALUDE_class_composition_l1628_162802

/-- Represents a child's response about the number of classmates -/
structure Response :=
  (boys : ℕ)
  (girls : ℕ)

/-- Checks if a response is valid according to the problem conditions -/
def is_valid_response (actual_boys actual_girls : ℕ) (r : Response) : Prop :=
  (r.boys = actual_boys ∧ (r.girls = actual_girls + 2 ∨ r.girls = actual_girls - 2)) ∨
  (r.girls = actual_girls ∧ (r.boys = actual_boys + 2 ∨ r.boys = actual_boys - 2))

/-- The main theorem stating the correct number of boys and girls in the class -/
theorem class_composition :
  ∃ (actual_boys actual_girls : ℕ),
    actual_boys = 15 ∧
    actual_girls = 12 ∧
    is_valid_response actual_boys actual_girls ⟨13, 11⟩ ∧
    is_valid_response actual_boys actual_girls ⟨17, 11⟩ ∧
    is_valid_response actual_boys actual_girls ⟨14, 14⟩ :=
  sorry

end NUMINAMATH_CALUDE_class_composition_l1628_162802


namespace NUMINAMATH_CALUDE_ababab_divisible_by_seven_l1628_162895

/-- Given two digits a and b, the function forms the number ababab -/
def formNumber (a b : ℕ) : ℕ := 
  100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b

/-- Theorem stating that for any two digits, the number formed as ababab is divisible by 7 -/
theorem ababab_divisible_by_seven (a b : ℕ) (ha : a < 10) (hb : b < 10) : 
  7 ∣ formNumber a b := by
  sorry

#eval formNumber 2 3  -- To check if the function works correctly

end NUMINAMATH_CALUDE_ababab_divisible_by_seven_l1628_162895


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1628_162854

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ (a b c d : ℕ),
    n = 1000 * a + 100 * b + 10 * c + d ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

def reverse_number (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

theorem unique_four_digit_number :
  ∀ n : ℕ, is_valid_number n → (reverse_number n = n + 7182) → n = 1909 :=
sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1628_162854


namespace NUMINAMATH_CALUDE_composite_4p_plus_1_l1628_162823

theorem composite_4p_plus_1 (p : ℕ) (h1 : p ≥ 5) (h2 : Nat.Prime p) (h3 : Nat.Prime (2 * p + 1)) :
  ¬(Nat.Prime (4 * p + 1)) :=
sorry

end NUMINAMATH_CALUDE_composite_4p_plus_1_l1628_162823


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1628_162865

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 6 ∧
  12 ∣ (427398 - x) ∧
  ∀ (y : ℕ), y < x → ¬(12 ∣ (427398 - y)) :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1628_162865


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1628_162832

/-- An isosceles triangle with specific measurements -/
structure IsoscelesTriangle where
  -- Base of the triangle
  base : ℝ
  -- Median from one of the equal sides
  median : ℝ
  -- Length of the equal sides
  side : ℝ
  -- The base is 4√2 cm
  base_eq : base = 4 * Real.sqrt 2
  -- The median is 5 cm
  median_eq : median = 5
  -- The triangle is isosceles (implied by the structure)

/-- 
Theorem: In an isosceles triangle with a base of 4√2 cm and a median of 5 cm 
from one of the equal sides, the length of the equal sides is 6 cm.
-/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) : t.side = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1628_162832


namespace NUMINAMATH_CALUDE_ab_max_and_sum_squares_min_l1628_162852

theorem ab_max_and_sum_squares_min (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 1) : 
  (∀ x y, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → a * b ≥ x * y) ∧
  (∀ x y, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → 4 * a^2 + b^2 ≤ 4 * x^2 + y^2) ∧
  a * b = 1/8 ∧
  4 * a^2 + b^2 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ab_max_and_sum_squares_min_l1628_162852


namespace NUMINAMATH_CALUDE_fraction_multiplication_simplification_l1628_162822

theorem fraction_multiplication_simplification :
  (270 : ℚ) / 18 * (7 : ℚ) / 210 * (12 : ℚ) / 4 = (3 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_simplification_l1628_162822


namespace NUMINAMATH_CALUDE_smallest_a_for_positive_integer_roots_l1628_162821

theorem smallest_a_for_positive_integer_roots : ∃ (a : ℕ),
  (∀ (x₁ x₂ : ℕ), x₁ * x₂ = 2022 ∧ x₁ + x₂ = a → x₁^2 - a*x₁ + 2022 = 0 ∧ x₂^2 - a*x₂ + 2022 = 0) ∧
  (∀ (b : ℕ), b < a →
    ¬∃ (y₁ y₂ : ℕ), y₁ * y₂ = 2022 ∧ y₁ + y₂ = b ∧ y₁^2 - b*y₁ + 2022 = 0 ∧ y₂^2 - b*y₂ + 2022 = 0) ∧
  a = 343 :=
by
  sorry

#check smallest_a_for_positive_integer_roots

end NUMINAMATH_CALUDE_smallest_a_for_positive_integer_roots_l1628_162821


namespace NUMINAMATH_CALUDE_calculation_proof_l1628_162831

theorem calculation_proof : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1628_162831


namespace NUMINAMATH_CALUDE_factorization_implies_k_value_l1628_162829

theorem factorization_implies_k_value (k : ℝ) :
  (∃ (a b c d e f : ℝ), ∀ x y : ℝ,
    x^3 + 3*x^2 - 2*x*y - k*x - 4*y = (a*x + b*y + c) * (d*x^2 + e*x*y + f*y)) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_factorization_implies_k_value_l1628_162829


namespace NUMINAMATH_CALUDE_fraction_division_l1628_162870

theorem fraction_division (a b : ℚ) (ha : a = 3) (hb : b = 4) :
  (1 / b) / (1 / a) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l1628_162870


namespace NUMINAMATH_CALUDE_bart_monday_surveys_l1628_162879

/-- The number of surveys Bart finished on Monday -/
def monday_surveys : ℕ := 3

/-- The amount earned per question in dollars -/
def earnings_per_question : ℚ := 1/5

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart finished on Tuesday -/
def tuesday_surveys : ℕ := 4

/-- The total amount Bart earned over Monday and Tuesday in dollars -/
def total_earnings : ℚ := 14

theorem bart_monday_surveys :
  monday_surveys * questions_per_survey * earnings_per_question +
  tuesday_surveys * questions_per_survey * earnings_per_question =
  total_earnings :=
sorry

end NUMINAMATH_CALUDE_bart_monday_surveys_l1628_162879


namespace NUMINAMATH_CALUDE_log_equation_two_roots_l1628_162864

-- Define the logarithmic equation
def log_equation (x a : ℝ) : Prop :=
  Real.log (2 * x) / Real.log (x + a) = 2

-- Define the conditions
def conditions (x a : ℝ) : Prop :=
  x > 0 ∧ x + a > 0 ∧ x + a ≠ 1

-- Theorem statement
theorem log_equation_two_roots :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    log_equation x₁ a ∧ log_equation x₂ a ∧ 
    conditions x₁ a ∧ conditions x₂ a) ↔ 
  (a > 0 ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_log_equation_two_roots_l1628_162864


namespace NUMINAMATH_CALUDE_more_pups_than_adults_l1628_162867

def num_huskies : ℕ := 5
def num_pitbulls : ℕ := 2
def num_golden_retrievers : ℕ := 4

def pups_per_husky : ℕ := 3
def pups_per_pitbull : ℕ := 3
def pups_per_golden_retriever : ℕ := pups_per_husky + 2

def total_adult_dogs : ℕ := num_huskies + num_pitbulls + num_golden_retrievers

def total_pups : ℕ := 
  num_huskies * pups_per_husky + 
  num_pitbulls * pups_per_pitbull + 
  num_golden_retrievers * pups_per_golden_retriever

theorem more_pups_than_adults : total_pups - total_adult_dogs = 30 := by
  sorry

end NUMINAMATH_CALUDE_more_pups_than_adults_l1628_162867


namespace NUMINAMATH_CALUDE_max_value_of_A_l1628_162898

theorem max_value_of_A (A B : ℕ) (h1 : A = 5 * 2 + B) (h2 : B < 5) : A ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_A_l1628_162898


namespace NUMINAMATH_CALUDE_condition_relationship_l1628_162866

theorem condition_relationship (p q : Prop) 
  (h : (¬p → ¬q) ∧ ¬(¬q → ¬p)) : 
  (p → q) ∧ ¬(q → p) :=
sorry

end NUMINAMATH_CALUDE_condition_relationship_l1628_162866


namespace NUMINAMATH_CALUDE_right_triangle_area_l1628_162843

/-- 
Given a right triangle with hypotenuse c, where the projection of the right angle 
vertex onto the hypotenuse divides it into two segments x and (c-x) such that 
(c-x)/x = x/c, the area of the triangle is (c^2 * sqrt(sqrt(5) - 2)) / 2.
-/
theorem right_triangle_area (c : ℝ) (h : c > 0) : 
  ∃ x : ℝ, 0 < x ∧ x < c ∧ (c - x) / x = x / c ∧ 
  (c^2 * Real.sqrt (Real.sqrt 5 - 2)) / 2 = 
  (1 / 2) * c * Real.sqrt (c * x - x^2) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1628_162843


namespace NUMINAMATH_CALUDE_equation_solution_l1628_162851

theorem equation_solution (y : ℕ) : 9^10 + 9^10 + 9^10 = 3^y → y = 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1628_162851


namespace NUMINAMATH_CALUDE_angle_double_supplement_is_120_l1628_162881

-- Define the angle measure
def angle_measure : ℝ → Prop := λ x => 
  -- The angle measure is double its supplement
  x = 2 * (180 - x) ∧ 
  -- The angle measure is positive and less than or equal to 180
  0 < x ∧ x ≤ 180

-- Theorem statement
theorem angle_double_supplement_is_120 : 
  ∃ x : ℝ, angle_measure x ∧ x = 120 :=
sorry

end NUMINAMATH_CALUDE_angle_double_supplement_is_120_l1628_162881


namespace NUMINAMATH_CALUDE_machine_speed_ratio_l1628_162837

def machine_a_rate (parts_a : ℕ) (time_a : ℕ) : ℚ := parts_a / time_a
def machine_b_rate (parts_b : ℕ) (time_b : ℕ) : ℚ := parts_b / time_b

theorem machine_speed_ratio :
  let parts_a_100 : ℕ := 100
  let time_a_100 : ℕ := 40
  let parts_a_50 : ℕ := 50
  let time_a_50 : ℕ := 10
  let parts_b : ℕ := 100
  let time_b : ℕ := 40
  machine_a_rate parts_a_100 time_a_100 = machine_b_rate parts_b time_b →
  machine_a_rate parts_a_50 time_a_50 / machine_b_rate parts_b time_b = 2 := by
sorry

end NUMINAMATH_CALUDE_machine_speed_ratio_l1628_162837


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l1628_162857

theorem roots_sum_and_product (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∀ x, x^2 - p*x - 2*q = 0 ↔ x = p ∨ x = q) :
  p + q = p ∧ p * q = -2*q := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l1628_162857


namespace NUMINAMATH_CALUDE_smallest_overlap_coffee_tea_l1628_162853

/-- The smallest possible percentage of adults who drink both coffee and tea,
    given that 50% drink coffee and 60% drink tea. -/
theorem smallest_overlap_coffee_tea : ℝ :=
  let coffee_drinkers : ℝ := 50
  let tea_drinkers : ℝ := 60
  let total_percentage : ℝ := 100
  min (coffee_drinkers + tea_drinkers - total_percentage) coffee_drinkers

end NUMINAMATH_CALUDE_smallest_overlap_coffee_tea_l1628_162853


namespace NUMINAMATH_CALUDE_inner_rectangle_length_l1628_162817

/-- Represents the dimensions of a rectangular region -/
structure Region where
  length : ℝ
  width : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Represents the floor layout with three regions -/
structure FloorLayout where
  inner : Region
  middle : Region
  outer : Region

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem inner_rectangle_length (layout : FloorLayout) : 
  layout.inner.width = 2 →
  layout.middle.length = layout.inner.length + 2 →
  layout.middle.width = layout.inner.width + 2 →
  layout.outer.length = layout.middle.length + 2 →
  layout.outer.width = layout.middle.width + 2 →
  isArithmeticProgression (area layout.inner) (area layout.middle) (area layout.outer) →
  layout.inner.length = 8 := by
  sorry

#check inner_rectangle_length

end NUMINAMATH_CALUDE_inner_rectangle_length_l1628_162817


namespace NUMINAMATH_CALUDE_seating_arrangement_probability_l1628_162862

/-- Represents the number of delegates --/
def total_delegates : ℕ := 12

/-- Represents the number of countries --/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country --/
def delegates_per_country : ℕ := 4

/-- Calculates the probability of the seating arrangement --/
noncomputable def seating_probability : ℚ :=
  409 / 500

/-- Theorem stating the probability of the specific seating arrangement --/
theorem seating_arrangement_probability :
  let total_arrangements := (total_delegates.factorial) / (delegates_per_country.factorial ^ num_countries)
  let favorable_arrangements := total_arrangements - (num_countries * total_delegates * 
    ((total_delegates - delegates_per_country).factorial / (delegates_per_country.factorial ^ (num_countries - 1))) -
    (num_countries * (num_countries - 1) / 2 * total_delegates * (total_delegates - 2)) +
    (total_delegates * (num_countries - 1)))
  (favorable_arrangements : ℚ) / total_arrangements = seating_probability :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_probability_l1628_162862


namespace NUMINAMATH_CALUDE_average_of_set_l1628_162849

theorem average_of_set (S : Finset ℕ) (n : ℕ) (h_nonempty : S.Nonempty) :
  (∃ (max min : ℕ),
    max ∈ S ∧ min ∈ S ∧
    (∀ x ∈ S, x ≤ max) ∧
    (∀ x ∈ S, min ≤ x) ∧
    (S.sum id - max) / (S.card - 1) = 32 ∧
    (S.sum id - max - min) / (S.card - 2) = 35 ∧
    (S.sum id - min) / (S.card - 1) = 40 ∧
    max = min + 72) →
  S.sum id / S.card = 368 / 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_set_l1628_162849


namespace NUMINAMATH_CALUDE_longest_collection_pages_l1628_162803

/-- Represents the number of pages per inch for a book collection -/
structure PagesPerInch where
  value : ℕ

/-- Represents the height of a book collection in inches -/
structure CollectionHeight where
  value : ℕ

/-- Calculates the total number of pages in a collection -/
def total_pages (ppi : PagesPerInch) (height : CollectionHeight) : ℕ :=
  ppi.value * height.value

/-- Represents Miles's book collection -/
def miles_collection : PagesPerInch × CollectionHeight :=
  ({ value := 5 }, { value := 240 })

/-- Represents Daphne's book collection -/
def daphne_collection : PagesPerInch × CollectionHeight :=
  ({ value := 50 }, { value := 25 })

theorem longest_collection_pages : 
  max (total_pages miles_collection.1 miles_collection.2)
      (total_pages daphne_collection.1 daphne_collection.2) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_longest_collection_pages_l1628_162803


namespace NUMINAMATH_CALUDE_gold_bars_per_row_l1628_162859

/-- Represents the arrangement of gold bars in a safe -/
structure GoldSafe where
  rows : Nat
  totalWorth : Nat
  barValue : Nat

/-- Calculates the number of gold bars per row in a safe -/
def barsPerRow (safe : GoldSafe) : Nat :=
  (safe.totalWorth / safe.barValue) / safe.rows

/-- Theorem: If a safe has 4 rows, total worth of $1,600,000, and each bar is worth $40,000,
    then there are 10 gold bars in each row -/
theorem gold_bars_per_row :
  let safe : GoldSafe := { rows := 4, totalWorth := 1600000, barValue := 40000 }
  barsPerRow safe = 10 := by
  sorry


end NUMINAMATH_CALUDE_gold_bars_per_row_l1628_162859


namespace NUMINAMATH_CALUDE_set_operations_and_subset_condition_l1628_162880

def A : Set ℝ := {x | -4 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

theorem set_operations_and_subset_condition :
  (∀ x, x ∈ (A ∪ B 1) ↔ -4 < x ∧ x ≤ 3) ∧
  (∀ x, x ∈ (A ∩ (Set.univ \ B 1)) ↔ -4 < x ∧ x < 0) ∧
  (∀ a, B a ⊆ A ↔ -3 < a ∧ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_condition_l1628_162880


namespace NUMINAMATH_CALUDE_inequality_proof_l1628_162813

theorem inequality_proof (x y z : ℝ) 
  (h1 : x^2 + y*z ≠ 0) (h2 : y^2 + z*x ≠ 0) (h3 : z^2 + x*y ≠ 0) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1628_162813


namespace NUMINAMATH_CALUDE_gravel_weight_in_specific_mixture_l1628_162882

/-- A cement mixture with sand, water, and gravel -/
structure CementMixture where
  total_weight : ℝ
  sand_fraction : ℝ
  water_fraction : ℝ

/-- Calculate the weight of gravel in a cement mixture -/
def gravel_weight (m : CementMixture) : ℝ :=
  m.total_weight * (1 - m.sand_fraction - m.water_fraction)

/-- Theorem stating the weight of gravel in the specific mixture -/
theorem gravel_weight_in_specific_mixture :
  let m : CementMixture := {
    total_weight := 120,
    sand_fraction := 1/5,
    water_fraction := 3/4
  }
  gravel_weight m = 6 := by
  sorry

end NUMINAMATH_CALUDE_gravel_weight_in_specific_mixture_l1628_162882


namespace NUMINAMATH_CALUDE_inequality_implication_l1628_162810

theorem inequality_implication (a b : ℝ) : a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1628_162810


namespace NUMINAMATH_CALUDE_jacket_cost_price_l1628_162825

theorem jacket_cost_price (original_price discount profit : ℝ) 
  (h1 : original_price = 500)
  (h2 : discount = 0.3)
  (h3 : profit = 50) :
  original_price * (1 - discount) = 300 + profit := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_price_l1628_162825


namespace NUMINAMATH_CALUDE_divisibility_of_sum_and_powers_l1628_162833

theorem divisibility_of_sum_and_powers (a b c : ℤ) 
  (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_and_powers_l1628_162833


namespace NUMINAMATH_CALUDE_furniture_reimbursement_l1628_162893

/-- Calculates the reimbursement amount for an overcharged furniture purchase -/
theorem furniture_reimbursement
  (total_paid : ℕ)
  (num_pieces : ℕ)
  (cost_per_piece : ℕ)
  (h1 : total_paid = 20700)
  (h2 : num_pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (num_pieces * cost_per_piece) = 600 := by
sorry

end NUMINAMATH_CALUDE_furniture_reimbursement_l1628_162893


namespace NUMINAMATH_CALUDE_edges_after_intersection_l1628_162887

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- Represents the result of intersecting a polyhedron with planes -/
def intersect_with_planes (Q : ConvexPolyhedron) (num_planes : ℕ) : ℕ := sorry

/-- Theorem: The number of edges after intersection is 450 -/
theorem edges_after_intersection (Q : ConvexPolyhedron) (h1 : Q.edges = 150) :
  intersect_with_planes Q Q.vertices = 450 := by sorry

end NUMINAMATH_CALUDE_edges_after_intersection_l1628_162887


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1628_162818

theorem triangle_angle_calculation (A B C : ℝ) : 
  A = 60 →                 -- A is 60 degrees
  B = 2 * C →              -- B is two times as big as C
  A + B + C = 180 →        -- Sum of angles in a triangle is 180 degrees
  B = 80 := by             -- Prove that B is 80 degrees
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1628_162818


namespace NUMINAMATH_CALUDE_variance_of_dataset_l1628_162835

def dataset : List ℝ := [5, 7, 7, 8, 10, 11]

/-- The variance of the dataset [5, 7, 7, 8, 10, 11] is 4 -/
theorem variance_of_dataset : 
  let n : ℝ := dataset.length
  let mean : ℝ := (dataset.sum) / n
  let variance : ℝ := (dataset.map (λ x => (x - mean)^2)).sum / n
  variance = 4 := by sorry

end NUMINAMATH_CALUDE_variance_of_dataset_l1628_162835


namespace NUMINAMATH_CALUDE_exterior_angle_regular_pentagon_exterior_angle_regular_pentagon_proof_l1628_162876

/-- The size of an exterior angle of a regular pentagon is 72 degrees. -/
theorem exterior_angle_regular_pentagon : ℝ :=
  72

/-- The number of sides in a pentagon. -/
def pentagon_sides : ℕ := 5

/-- The sum of exterior angles of any polygon in degrees. -/
def sum_exterior_angles : ℝ := 360

/-- Theorem: The size of an exterior angle of a regular pentagon is 72 degrees. -/
theorem exterior_angle_regular_pentagon_proof :
  exterior_angle_regular_pentagon = sum_exterior_angles / pentagon_sides :=
by sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_pentagon_exterior_angle_regular_pentagon_proof_l1628_162876


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l1628_162878

def matrix_not_invertible (a b c : ℝ) : Prop :=
  ∀ k : ℝ, Matrix.det
    !![a + k, b + k, c + k;
       b + k, c + k, a + k;
       c + k, a + k, b + k] = 0

theorem matrix_sum_theorem (a b c : ℝ) :
  matrix_not_invertible a b c →
  (a / (b + c) + b / (a + c) + c / (a + b) = -3 ∨
   a / (b + c) + b / (a + c) + c / (a + b) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l1628_162878


namespace NUMINAMATH_CALUDE_carpenters_completion_time_l1628_162877

def carpenter1_rate : ℚ := 1 / 5
def carpenter2_rate : ℚ := 1 / 5
def combined_rate : ℚ := carpenter1_rate + carpenter2_rate
def job_completion : ℚ := 1

theorem carpenters_completion_time :
  ∃ (time : ℚ), time * combined_rate = job_completion ∧ time = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_carpenters_completion_time_l1628_162877


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l1628_162863

theorem min_sum_reciprocals (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) :
  1/m + 1/n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l1628_162863


namespace NUMINAMATH_CALUDE_hallie_tuesday_hours_l1628_162860

/-- Calculates the number of hours Hallie worked on Tuesday given her earnings and tips -/
def hours_worked_tuesday (hourly_rate : ℚ) (monday_hours : ℚ) (monday_tips : ℚ) 
  (tuesday_tips : ℚ) (wednesday_hours : ℚ) (wednesday_tips : ℚ) (total_earnings : ℚ) : ℚ :=
  let monday_earnings := hourly_rate * monday_hours + monday_tips
  let wednesday_earnings := hourly_rate * wednesday_hours + wednesday_tips
  let tuesday_earnings := total_earnings - monday_earnings - wednesday_earnings
  let tuesday_wage_earnings := tuesday_earnings - tuesday_tips
  tuesday_wage_earnings / hourly_rate

theorem hallie_tuesday_hours :
  hours_worked_tuesday 10 7 18 12 7 20 240 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hallie_tuesday_hours_l1628_162860


namespace NUMINAMATH_CALUDE_planes_perpendicular_l1628_162869

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : perp_line_plane m α) 
  (h4 : perp_line_plane n β) 
  (h5 : perp_line_line m n) : 
  perp_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l1628_162869


namespace NUMINAMATH_CALUDE_equation_solution_l1628_162836

theorem equation_solution (x : ℂ) : 
  (x - 2)^6 + (x - 6)^6 = 64 ↔ x = 4 + Complex.I * Real.sqrt 2 ∨ x = 4 - Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1628_162836


namespace NUMINAMATH_CALUDE_parabola_point_ordinate_l1628_162885

theorem parabola_point_ordinate (x y : ℝ) : 
  y^2 = 8*x →                  -- Point M(x, y) is on the parabola y^2 = 8x
  (x - 2)^2 + y^2 = 4^2 →      -- Distance from M to focus (2, 0) is 4
  y = 4 ∨ y = -4 :=            -- The ordinate of M is either 4 or -4
by sorry

end NUMINAMATH_CALUDE_parabola_point_ordinate_l1628_162885


namespace NUMINAMATH_CALUDE_mean_diesel_cost_l1628_162897

def diesel_rates : List ℝ := [1.2, 1.3, 1.8, 2.1]

theorem mean_diesel_cost (rates : List ℝ) (h : rates = diesel_rates) :
  (rates.sum / rates.length : ℝ) = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_mean_diesel_cost_l1628_162897


namespace NUMINAMATH_CALUDE_angle_supplement_complement_difference_l1628_162889

theorem angle_supplement_complement_difference (α : ℝ) : (180 - α) - (90 - α) = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_complement_difference_l1628_162889


namespace NUMINAMATH_CALUDE_angle_A_is_90_l1628_162830

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = 180

-- Define the specific conditions of our triangle
def our_triangle (t : Triangle) : Prop :=
  is_valid_triangle t ∧ t.C = 2 * t.B ∧ t.B = 30

-- Theorem statement
theorem angle_A_is_90 (t : Triangle) (h : our_triangle t) : t.A = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_90_l1628_162830


namespace NUMINAMATH_CALUDE_power_division_equality_l1628_162848

theorem power_division_equality : (19 : ℕ) ^ 11 / (19 : ℕ) ^ 5 = 47045881 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l1628_162848


namespace NUMINAMATH_CALUDE_function_divides_property_l1628_162801

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem function_divides_property 
  (f : ℤ → ℕ+) 
  (h : ∀ m n : ℤ, divides (f (m - n)) (f m - f n)) :
  ∀ n m : ℤ, f n ≤ f m → divides (f n) (f m) := by
  sorry

end NUMINAMATH_CALUDE_function_divides_property_l1628_162801


namespace NUMINAMATH_CALUDE_rational_roots_imply_rational_roots_l1628_162826

theorem rational_roots_imply_rational_roots (c : ℝ) (p q : ℚ) :
  (p^2 - p + c = 0) → (q^2 - q + c = 0) →
  ∃ (r s : ℚ), r^2 + p*r - q = 0 ∧ s^2 + p*s - q = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_roots_imply_rational_roots_l1628_162826


namespace NUMINAMATH_CALUDE_root_implies_sum_l1628_162811

theorem root_implies_sum (c d : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2)^3 + c * (Complex.I * Real.sqrt 2 + 2) + d = 0 → 
  c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_root_implies_sum_l1628_162811
