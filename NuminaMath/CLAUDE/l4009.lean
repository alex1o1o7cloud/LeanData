import Mathlib

namespace max_value_of_x_plus_inverse_l4009_400961

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (max : ℝ), max = Real.sqrt 15 ∧ x + 1/x ≤ max ∧ ∃ (y : ℝ), 13 = y^2 + 1/y^2 ∧ y + 1/y = max :=
by sorry

end max_value_of_x_plus_inverse_l4009_400961


namespace fifteen_is_counterexample_l4009_400981

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_counterexample (n : ℕ) : Prop :=
  ¬(is_prime n) ∧ ¬(is_prime (n - 3))

theorem fifteen_is_counterexample :
  is_counterexample 15 :=
sorry

end fifteen_is_counterexample_l4009_400981


namespace geometric_series_ratio_l4009_400993

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4) / (1 - r)) → r = 1/2 := by
  sorry

end geometric_series_ratio_l4009_400993


namespace race_distance_l4009_400916

theorem race_distance (p q : ℝ) (d : ℝ) : 
  p = 1.2 * q →  -- p is 20% faster than q
  d = q * (d + 50) / p →  -- race ends in a tie
  d + 50 = 300 :=  -- p runs 300 meters
by sorry

end race_distance_l4009_400916


namespace polygon_sides_l4009_400997

theorem polygon_sides (sum_interior_angles : ℝ) (h : sum_interior_angles = 1680) :
  ∃ (n : ℕ), n = 12 ∧ (n - 2) * 180 > sum_interior_angles ∧ (n - 2) * 180 ≤ sum_interior_angles + 180 := by
  sorry

end polygon_sides_l4009_400997


namespace triangle_proof_l4009_400925

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_proof (ABC : Triangle) 
  (h1 : ABC.B.sin = 1/3)
  (h2 : ABC.a^2 - ABC.b^2 + ABC.c^2 = 2 ∨ ABC.a * ABC.c * ABC.B.cos = -1)
  (h3 : ABC.A.sin * ABC.C.sin = Real.sqrt 2 / 3) :
  (ABC.a * ABC.c = 3 * Real.sqrt 2 / 4) ∧ 
  (ABC.b = 1/2) := by
sorry

end triangle_proof_l4009_400925


namespace triangle_angle_c_l4009_400984

theorem triangle_angle_c (A B C : ℝ) (h1 : A + B = 110) : C = 70 := by
  sorry

end triangle_angle_c_l4009_400984


namespace seconds_in_misfortune_day_l4009_400975

/-- The number of minutes in a day on the island of Misfortune -/
def minutes_per_day : ℕ := 77

/-- The number of seconds in a minute on the island of Misfortune -/
def seconds_per_minute : ℕ := 91

/-- Theorem: The number of seconds in a day on the island of Misfortune is 1001 -/
theorem seconds_in_misfortune_day : 
  minutes_per_day * seconds_per_minute = 1001 := by
  sorry

end seconds_in_misfortune_day_l4009_400975


namespace line_translation_proof_l4009_400933

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given distance -/
def translateLine (l : Line) (distance : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + distance }

theorem line_translation_proof :
  let originalLine : Line := { slope := 2, intercept := -3 }
  let translatedLine := translateLine originalLine 6
  translatedLine = { slope := 2, intercept := 3 } := by
sorry

end line_translation_proof_l4009_400933


namespace percentage_relation_l4009_400962

theorem percentage_relation (x y : ℝ) (h : 0.15 * x = 0.2 * y) : y = 0.75 * x := by
  sorry

end percentage_relation_l4009_400962


namespace greene_nursery_white_roses_l4009_400983

/-- The number of white roses at Greene Nursery -/
def white_roses : ℕ := 6284 - (1491 + 3025)

/-- Theorem stating the number of white roses at Greene Nursery -/
theorem greene_nursery_white_roses :
  white_roses = 1768 :=
by sorry

end greene_nursery_white_roses_l4009_400983


namespace opposite_of_negative_l4009_400971

theorem opposite_of_negative (a : ℝ) : -(- a) = a := by sorry

end opposite_of_negative_l4009_400971


namespace proportion_problem_l4009_400976

theorem proportion_problem (y : ℝ) : (0.75 / 2 = 3 / y) → y = 8 := by
  sorry

end proportion_problem_l4009_400976


namespace decimal_0_03_is_3_percent_l4009_400946

/-- Converts a decimal fraction to a percentage -/
def decimal_to_percentage (d : ℝ) : ℝ := d * 100

/-- The decimal fraction we're working with -/
def given_decimal : ℝ := 0.03

/-- Theorem: The percentage equivalent of 0.03 is 3% -/
theorem decimal_0_03_is_3_percent :
  decimal_to_percentage given_decimal = 3 := by
  sorry

end decimal_0_03_is_3_percent_l4009_400946


namespace cosine_amplitude_l4009_400987

/-- Given a cosine function y = a * cos(bx) where a > 0 and b > 0,
    if the maximum value is 3 and the minimum value is -3, then a = 3 -/
theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, a * Real.cos (b * x) ≤ 3) 
  (hmin : ∀ x, a * Real.cos (b * x) ≥ -3)
  (hreach_max : ∃ x, a * Real.cos (b * x) = 3)
  (hreach_min : ∃ x, a * Real.cos (b * x) = -3) : 
  a = 3 := by sorry

end cosine_amplitude_l4009_400987


namespace factory_composition_diagram_l4009_400926

/-- Represents different types of diagrams --/
inductive Diagram
  | ProgramFlowchart
  | ProcessFlow
  | KnowledgeStructure
  | OrganizationalStructure

/-- Represents the purpose of a diagram --/
inductive DiagramPurpose
  | RepresentComposition
  | RepresentProcedures
  | RepresentKnowledge

/-- Associates a diagram type with its primary purpose --/
def diagramPurpose (d : Diagram) : DiagramPurpose :=
  match d with
  | Diagram.ProgramFlowchart => DiagramPurpose.RepresentProcedures
  | Diagram.ProcessFlow => DiagramPurpose.RepresentProcedures
  | Diagram.KnowledgeStructure => DiagramPurpose.RepresentKnowledge
  | Diagram.OrganizationalStructure => DiagramPurpose.RepresentComposition

/-- The theorem stating that the Organizational Structure Diagram 
    is used to represent the composition of a factory --/
theorem factory_composition_diagram :
  diagramPurpose Diagram.OrganizationalStructure = DiagramPurpose.RepresentComposition :=
by sorry


end factory_composition_diagram_l4009_400926


namespace power_six_2045_mod_13_l4009_400988

theorem power_six_2045_mod_13 : 6^2045 ≡ 2 [ZMOD 13] := by sorry

end power_six_2045_mod_13_l4009_400988


namespace remaining_distance_l4009_400936

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 384) :
  total_distance - driven_distance = 816 := by
  sorry

end remaining_distance_l4009_400936


namespace final_values_l4009_400947

def sequence_operations (a b c : ℕ) : ℕ × ℕ × ℕ :=
  let a' := b
  let b' := c
  let c' := a'
  (a', b', c')

theorem final_values :
  sequence_operations 10 20 30 = (20, 30, 20) := by sorry

end final_values_l4009_400947


namespace arithmetic_calculation_l4009_400910

theorem arithmetic_calculation : 6 / (-3) + 2^2 * (1 - 4) = -14 := by
  sorry

end arithmetic_calculation_l4009_400910


namespace selling_price_equation_l4009_400950

/-- Represents the selling price of pants in a store -/
def selling_price (X : ℝ) : ℝ :=
  let initial_price := X
  let discount_rate := 0.1
  let bulk_discount := 5
  let markup_rate := 0.25
  let discounted_price := initial_price * (1 - discount_rate)
  let final_purchase_cost := discounted_price - bulk_discount
  let marked_up_price := final_purchase_cost * (1 + markup_rate)
  marked_up_price

/-- Theorem stating the relationship between initial purchase price and selling price -/
theorem selling_price_equation (X : ℝ) :
  selling_price X = 1.125 * X - 6.25 := by
  sorry

end selling_price_equation_l4009_400950


namespace algebraic_expression_simplification_l4009_400966

theorem algebraic_expression_simplification (a : ℝ) (h : a = Real.sqrt 2) :
  a / (a^2 - 2*a + 1) / (1 + 1 / (a - 1)) = Real.sqrt 2 + 1 := by
  sorry

end algebraic_expression_simplification_l4009_400966


namespace cubic_roots_sum_l4009_400964

theorem cubic_roots_sum (a b c : ℝ) (h1 : a < b) (h2 : b < c)
  (ha : a^3 - 3*a + 1 = 0) (hb : b^3 - 3*b + 1 = 0) (hc : c^3 - 3*c + 1 = 0) :
  1 / (a^2 + b) + 1 / (b^2 + c) + 1 / (c^2 + a) = 3 := by
  sorry

end cubic_roots_sum_l4009_400964


namespace monotonic_increase_interval_l4009_400992

/-- Given a function f with period π and its left-translated version g, 
    prove the interval of monotonic increase for g. -/
theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_f_def : ∀ x, f x = Real.sin (ω * x - π / 4))
  (h_f_period : ∀ x, f (x + π) = f x)
  (g : ℝ → ℝ)
  (h_g_def : ∀ x, g x = f (x + π / 4)) :
  ∀ k : ℤ, StrictMonoOn g (Set.Icc (-3 * π / 8 + k * π) (π / 8 + k * π)) :=
sorry

end monotonic_increase_interval_l4009_400992


namespace largest_satisfying_n_l4009_400930

/-- A rectangle with sides parallel to the coordinate axes -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- Two rectangles are disjoint -/
def disjoint (r1 r2 : Rectangle) : Prop :=
  r1.x_max ≤ r2.x_min ∨ r2.x_max ≤ r1.x_min ∨
  r1.y_max ≤ r2.y_min ∨ r2.y_max ≤ r1.y_min

/-- Two rectangles have a common point -/
def have_common_point (r1 r2 : Rectangle) : Prop :=
  ¬(disjoint r1 r2)

/-- The property described in the problem -/
def satisfies_property (n : ℕ) : Prop :=
  ∃ (A B : Fin n → Rectangle),
    (∀ i : Fin n, disjoint (A i) (B i)) ∧
    (∀ i j : Fin n, i ≠ j → have_common_point (A i) (B j))

/-- The main theorem: The largest positive integer satisfying the property is 4 -/
theorem largest_satisfying_n :
  (∃ n : ℕ, n > 0 ∧ satisfies_property n) ∧
  (∀ n : ℕ, satisfies_property n → n ≤ 4) ∧
  satisfies_property 4 := by
  sorry

end largest_satisfying_n_l4009_400930


namespace a_equals_four_l4009_400965

theorem a_equals_four (a : ℝ) (h : a * 2 * (2^3) = 2^6) : a = 4 := by
  sorry

end a_equals_four_l4009_400965


namespace louise_pictures_l4009_400998

def total_pictures (vertical horizontal haphazard : ℕ) : ℕ :=
  vertical + horizontal + haphazard

theorem louise_pictures : 
  ∀ (vertical horizontal haphazard : ℕ),
    vertical = 10 →
    horizontal = vertical / 2 →
    haphazard = 5 →
    total_pictures vertical horizontal haphazard = 20 :=
by
  sorry

end louise_pictures_l4009_400998


namespace cream_cake_problem_l4009_400938

def creamPerCake : ℕ := 75
def totalCream : ℕ := 500
def totalCakes : ℕ := 50
def cakesPerBox : ℕ := 6

theorem cream_cake_problem :
  (totalCream / creamPerCake : ℕ) = 6 ∧
  (totalCakes + cakesPerBox - 1) / cakesPerBox = 9 := by
  sorry

end cream_cake_problem_l4009_400938


namespace unique_number_with_three_prime_factors_l4009_400941

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 9^n - 1 → 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 7 ∧ q ≠ 7 ∧ 
    x = 2^(Nat.log 2 x) * 7 * p * q) →
  x = 728 := by
sorry

end unique_number_with_three_prime_factors_l4009_400941


namespace radical_conjugate_sum_product_l4009_400960

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = 0 ∧
  (a + Real.sqrt b) * (a - Real.sqrt b) = 25 →
  a + b = -25 := by
sorry

end radical_conjugate_sum_product_l4009_400960


namespace square_area_on_xz_l4009_400937

/-- A right-angled triangle with squares on each side -/
structure RightTriangleWithSquares where
  /-- Length of side XZ -/
  x : ℝ
  /-- The sum of areas of squares on all sides is 500 -/
  area_sum : x^2 / 2 + x^2 + 5 * x^2 / 4 = 500

/-- The area of the square on side XZ is 2000/11 -/
theorem square_area_on_xz (t : RightTriangleWithSquares) : t.x^2 = 2000 / 11 := by
  sorry

#check square_area_on_xz

end square_area_on_xz_l4009_400937


namespace max_abs_diff_on_interval_l4009_400927

open Real

-- Define the functions
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the absolute difference function
def abs_diff (x : ℝ) : ℝ := |f x - g x|

-- State the theorem
theorem max_abs_diff_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc 0 1 → abs_diff x ≤ abs_diff c ∧
  abs_diff c = 4/27 :=
sorry

end max_abs_diff_on_interval_l4009_400927


namespace cecil_money_problem_l4009_400921

theorem cecil_money_problem (cecil : ℝ) (catherine : ℝ) (carmela : ℝ) 
  (h1 : catherine = 2 * cecil - 250)
  (h2 : carmela = 2 * cecil + 50)
  (h3 : cecil + catherine + carmela = 2800) :
  cecil = 600 := by
sorry

end cecil_money_problem_l4009_400921


namespace dads_nickels_l4009_400977

/-- The number of nickels Tim had initially -/
def initial_nickels : ℕ := 9

/-- The number of nickels Tim has now -/
def current_nickels : ℕ := 12

/-- The number of nickels Tim's dad gave him -/
def nickels_from_dad : ℕ := current_nickels - initial_nickels

theorem dads_nickels : nickels_from_dad = 3 := by
  sorry

end dads_nickels_l4009_400977


namespace sequence_closed_form_l4009_400939

def recurrence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = Real.sqrt ((a n + 2 - Real.sqrt (2 - a n)) / 2)

theorem sequence_closed_form (a : ℕ → ℝ) :
  recurrence a ∧ a 0 = Real.sqrt 2 / 2 →
  ∀ n, a n = Real.sqrt 2 * Real.cos (π / 4 + π / (12 * 2^n)) :=
sorry

end sequence_closed_form_l4009_400939


namespace not_coplanar_implies_no_intersection_l4009_400974

-- Define a point in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a line in 3D space as two points
def Line3D := Point3D × Point3D

-- Define a function to check if four points are coplanar
def are_coplanar (E F G H : Point3D) : Prop := sorry

-- Define a function to check if two lines intersect
def lines_intersect (l1 l2 : Line3D) : Prop := sorry

theorem not_coplanar_implies_no_intersection 
  (E F G H : Point3D) : 
  ¬(are_coplanar E F G H) → ¬(lines_intersect (E, F) (G, H)) := by
  sorry

end not_coplanar_implies_no_intersection_l4009_400974


namespace goldfish_problem_l4009_400913

/-- The number of goldfish that died -/
def goldfish_died (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem goldfish_problem :
  let initial : ℕ := 89
  let remaining : ℕ := 57
  goldfish_died initial remaining = 32 := by
sorry

end goldfish_problem_l4009_400913


namespace supermarket_spending_l4009_400942

theorem supermarket_spending (F : ℚ) : 
  (∃ (M : ℚ), 
    M = 150 ∧ 
    F * M + (1/3) * M + (1/10) * M + 10 = M) →
  F = 1/2 := by
sorry

end supermarket_spending_l4009_400942


namespace wickets_before_last_match_value_l4009_400990

/-- Represents the bowling statistics of a cricket player -/
structure BowlingStats where
  initial_average : ℝ
  initial_wickets : ℕ
  new_wickets : ℕ
  new_runs : ℕ
  average_decrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wickets_before_last_match (stats : BowlingStats) : ℕ :=
  stats.initial_wickets

/-- Theorem stating the number of wickets taken before the last match -/
theorem wickets_before_last_match_value (stats : BowlingStats) 
  (h1 : stats.initial_average = 12.4)
  (h2 : stats.new_wickets = 3)
  (h3 : stats.new_runs = 26)
  (h4 : stats.average_decrease = 0.4)
  (h5 : stats.initial_wickets = wickets_before_last_match stats) :
  wickets_before_last_match stats = 25 := by
  sorry

#eval wickets_before_last_match { 
  initial_average := 12.4, 
  initial_wickets := 25, 
  new_wickets := 3, 
  new_runs := 26, 
  average_decrease := 0.4 
}

end wickets_before_last_match_value_l4009_400990


namespace intersection_M_N_l4009_400931

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N : M ∩ N = Set.Ioo 1 3 := by sorry

end intersection_M_N_l4009_400931


namespace equation_solution_l4009_400909

theorem equation_solution : ∃ y : ℚ, y - 1/2 = 1/6 - 2/3 + 1/4 ∧ y = 1/4 := by
  sorry

end equation_solution_l4009_400909


namespace remainder_2019_pow_2018_mod_100_l4009_400972

theorem remainder_2019_pow_2018_mod_100 : 2019^2018 ≡ 41 [ZMOD 100] := by sorry

end remainder_2019_pow_2018_mod_100_l4009_400972


namespace remainder_mod_seven_l4009_400978

theorem remainder_mod_seven : (9^5 + 8^4 + 7^9) % 7 = 5 := by
  sorry

end remainder_mod_seven_l4009_400978


namespace bicycle_speed_l4009_400907

/-- Proves that given a 400 km trip where the first 100 km is traveled at speed v km/h
    and the remaining 300 km at 15 km/h, if the average speed for the entire trip is 16 km/h,
    then v = 20 km/h. -/
theorem bicycle_speed (v : ℝ) :
  v > 0 →
  (100 / v + 300 / 15 = 400 / 16) →
  v = 20 :=
by sorry

end bicycle_speed_l4009_400907


namespace notebook_distribution_l4009_400954

theorem notebook_distribution (total_notebooks : ℕ) (initial_students : ℕ) : 
  total_notebooks = 512 →
  total_notebooks = initial_students * (initial_students / 8) →
  (total_notebooks / (initial_students / 2) : ℕ) = 16 := by
  sorry

end notebook_distribution_l4009_400954


namespace barefoot_kids_l4009_400980

theorem barefoot_kids (total : ℕ) (socks : ℕ) (shoes : ℕ) (both : ℕ) 
  (h1 : total = 35)
  (h2 : socks = 18)
  (h3 : shoes = 15)
  (h4 : both = 8) :
  total - (socks + shoes - both) = 10 := by
sorry

end barefoot_kids_l4009_400980


namespace car_average_speed_l4009_400914

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 50) :
  (speed1 + speed2) / 2 = 70 := by
  sorry

end car_average_speed_l4009_400914


namespace circular_class_properties_l4009_400943

/-- Represents a circular seating arrangement of students -/
structure CircularClass where
  totalStudents : ℕ
  boyOppositePositions : (ℕ × ℕ)
  everyOtherIsBoy : Bool

/-- Calculates the number of boys in the class -/
def numberOfBoys (c : CircularClass) : ℕ :=
  c.totalStudents / 2

/-- Theorem stating the properties of the circular class -/
theorem circular_class_properties (c : CircularClass) 
  (h1 : c.boyOppositePositions = (10, 40))
  (h2 : c.everyOtherIsBoy = true) :
  c.totalStudents = 60 ∧ numberOfBoys c = 30 := by
  sorry

#check circular_class_properties

end circular_class_properties_l4009_400943


namespace binary_to_base5_conversion_l4009_400940

-- Define the binary number
def binary_num : ℕ := 0b101101

-- Define the base-5 number
def base5_num : ℕ := 140

-- Theorem statement
theorem binary_to_base5_conversion :
  (binary_num : ℕ).digits 5 = base5_num.digits 5 := by
  sorry

end binary_to_base5_conversion_l4009_400940


namespace A_power_93_l4009_400970

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_93 : A ^ 93 = A := by sorry

end A_power_93_l4009_400970


namespace system_solution_existence_l4009_400911

theorem system_solution_existence (b : ℝ) : 
  (∃ (a x y : ℝ), y = -b - x^2 ∧ x^2 + y^2 + 8*a^2 = 4 + 4*a*(x + y)) ↔ 
  b ≤ 2 * Real.sqrt 2 + 1/4 := by
sorry

end system_solution_existence_l4009_400911


namespace count_zeros_100_to_50_l4009_400995

/-- The number of zeros following the numeral one in the expanded form of 100^50 -/
def zeros_after_one_in_100_to_50 : ℕ := 100

/-- Theorem stating that the number of zeros following the numeral one
    in the expanded form of 100^50 is equal to 100 -/
theorem count_zeros_100_to_50 :
  zeros_after_one_in_100_to_50 = 100 := by sorry

end count_zeros_100_to_50_l4009_400995


namespace rectangle_area_l4009_400945

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 206) :
  L * B = 2520 := by
  sorry

end rectangle_area_l4009_400945


namespace min_cut_off_length_is_82_l4009_400986

/-- Represents the rope cutting problem with given constraints -/
def RopeCuttingProblem (total_length : ℕ) (piece_lengths : List ℕ) (max_pieces : ℕ) : Prop :=
  total_length = 89 ∧
  piece_lengths = [7, 3, 1] ∧
  max_pieces = 25

/-- The minimum length of rope that must be cut off -/
def MinCutOffLength (total_length : ℕ) (piece_lengths : List ℕ) (max_pieces : ℕ) : ℕ := 82

/-- Theorem stating the minimum cut-off length for the rope cutting problem -/
theorem min_cut_off_length_is_82
  (total_length : ℕ) (piece_lengths : List ℕ) (max_pieces : ℕ)
  (h : RopeCuttingProblem total_length piece_lengths max_pieces) :
  MinCutOffLength total_length piece_lengths max_pieces = 82 := by
  sorry

end min_cut_off_length_is_82_l4009_400986


namespace diophantine_equation_solutions_l4009_400955

theorem diophantine_equation_solutions
  (a b c : ℤ) 
  (d : ℕ) 
  (h_d : d = Int.gcd a b) 
  (h_div : c % d = 0) 
  (x₀ y₀ : ℤ) 
  (h_particular : a * x₀ + b * y₀ = c) :
  ∀ (x y : ℤ), 
    (a * x + b * y = c) ↔ 
    (∃ (k : ℤ), x = x₀ + k * (b / d) ∧ y = y₀ - k * (a / d)) :=
by sorry

end diophantine_equation_solutions_l4009_400955


namespace triplet_equality_l4009_400917

theorem triplet_equality (a b c : ℝ) :
  a * (b^2 + c) = c * (c + a * b) →
  b * (c^2 + a) = a * (a + b * c) →
  c * (a^2 + b) = b * (b + a * c) →
  a = b ∧ b = c :=
by sorry

end triplet_equality_l4009_400917


namespace work_completion_time_l4009_400952

/-- Given a work that can be completed by X in 40 days, prove that if X works for 8 days
    and Y finishes the remaining work in 32 days, then Y would take 40 days to complete
    the entire work alone. -/
theorem work_completion_time (x_total_days y_completion_days : ℕ) 
    (x_worked_days : ℕ) (h1 : x_total_days = 40) (h2 : x_worked_days = 8) 
    (h3 : y_completion_days = 32) : 
    (x_total_days : ℚ) * y_completion_days / (x_total_days - x_worked_days) = 40 :=
by sorry

end work_completion_time_l4009_400952


namespace arithmetic_sequence_range_l4009_400969

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_range (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 ≤ 7)
  (h_a6 : a 6 ≥ 9) :
  (a 10 > 11 ∧ ∀ M : ℝ, ∃ N : ℝ, a 10 > N ∧ N > M) :=
sorry

end arithmetic_sequence_range_l4009_400969


namespace range_of_a_l4009_400929

-- Define the conditions
def p (x : ℝ) : Prop := |x - 2| < 3
def q (x a : ℝ) : Prop := 0 < x ∧ x < a

-- State the theorem
theorem range_of_a :
  (∀ x a : ℝ, q x a → p x) ∧ 
  (∃ x a : ℝ, p x ∧ ¬(q x a)) →
  ∀ a : ℝ, (∃ x : ℝ, q x a) ↔ (0 < a ∧ a ≤ 5) :=
sorry

end range_of_a_l4009_400929


namespace managers_in_game_l4009_400919

/-- The number of managers participating in a volleyball game --/
def num_managers (total_teams : ℕ) (people_per_team : ℕ) (num_employees : ℕ) : ℕ :=
  total_teams * people_per_team - num_employees

/-- Theorem stating that the number of managers in the game is 3 --/
theorem managers_in_game :
  num_managers 3 2 3 = 3 := by
  sorry

end managers_in_game_l4009_400919


namespace correct_product_l4009_400904

theorem correct_product (a b : ℚ) (a_int b_int : ℕ) (result : ℕ) : 
  a = 0.125 →
  b = 5.12 →
  a_int = 125 →
  b_int = 512 →
  result = 64000 →
  a_int * b_int = result →
  a * b = 0.64 := by
sorry

end correct_product_l4009_400904


namespace percentage_problem_l4009_400924

theorem percentage_problem (P : ℝ) :
  (0.15 * 0.30 * (P / 100) * 4800 = 108) → P = 50 := by
  sorry

end percentage_problem_l4009_400924


namespace class_size_l4009_400959

theorem class_size (n : ℕ) (h1 : n < 50) (h2 : n % 8 = 5) (h3 : n % 6 = 3) : n = 21 ∨ n = 45 := by
  sorry

end class_size_l4009_400959


namespace marble_remainder_l4009_400979

theorem marble_remainder (r p : ℕ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 7) : 
  (r + p) % 8 = 4 := by
sorry

end marble_remainder_l4009_400979


namespace no_snow_probability_l4009_400908

theorem no_snow_probability (p : ℝ) (h : p = 3/4) :
  (1 - p)^3 = 1/64 := by
  sorry

end no_snow_probability_l4009_400908


namespace dividing_line_theorem_l4009_400957

/-- A configuration of six unit squares in two rows of three in the coordinate plane -/
structure SquareGrid :=
  (width : ℕ := 3)
  (height : ℕ := 2)

/-- A line extending from (2,0) to (k,k) -/
structure DividingLine :=
  (k : ℝ)

/-- The area above and below the line formed by the DividingLine -/
def areas (grid : SquareGrid) (line : DividingLine) : ℝ × ℝ :=
  sorry

/-- Theorem stating that k = 4 divides the grid such that the area above is twice the area below -/
theorem dividing_line_theorem (grid : SquareGrid) :
  ∃ (line : DividingLine), 
    let (area_below, area_above) := areas grid line
    line.k = 4 ∧ area_above = 2 * area_below := by sorry

end dividing_line_theorem_l4009_400957


namespace sum_first_102_remainder_l4009_400906

theorem sum_first_102_remainder (n : Nat) (h : n = 102) : 
  (n * (n + 1) / 2) % 5250 = 3 := by
  sorry

end sum_first_102_remainder_l4009_400906


namespace intersection_implies_m_equals_one_l4009_400953

def A (m : ℝ) : Set ℝ := {0, m}
def B : Set ℝ := {1, 2}

theorem intersection_implies_m_equals_one (m : ℝ) :
  A m ∩ B = {1} → m = 1 := by
  sorry

end intersection_implies_m_equals_one_l4009_400953


namespace ten_point_circle_triangles_l4009_400999

/-- A circle with 10 points and chords connecting every pair of points. -/
structure CircleWithChords where
  num_points : ℕ
  no_triple_intersections : Bool

/-- The number of triangles formed by chord intersections inside the circle. -/
def num_triangles (c : CircleWithChords) : ℕ := sorry

/-- Theorem stating that the number of triangles is 210 for a circle with 10 points. -/
theorem ten_point_circle_triangles (c : CircleWithChords) : 
  c.num_points = 10 → c.no_triple_intersections → num_triangles c = 210 := by
  sorry

end ten_point_circle_triangles_l4009_400999


namespace comparison_of_powers_l4009_400901

theorem comparison_of_powers (a b c : ℕ) : 
  a = 81^31 → b = 27^41 → c = 9^61 → a > b ∧ b > c :=
by sorry

end comparison_of_powers_l4009_400901


namespace opposite_and_absolute_value_l4009_400991

theorem opposite_and_absolute_value (x y : ℤ) :
  (- x = 3 ∧ |y| = 5) → (x + y = 2 ∨ x + y = -8) :=
by sorry

end opposite_and_absolute_value_l4009_400991


namespace orange_profit_maximization_l4009_400932

/-- Represents the cost and selling prices of oranges --/
structure OrangePrices where
  cost_a : ℝ
  sell_a : ℝ
  cost_b : ℝ
  sell_b : ℝ

/-- Represents a purchasing plan for oranges --/
structure PurchasePlan where
  kg_a : ℕ
  kg_b : ℕ

/-- Calculates the total cost of a purchase plan --/
def total_cost (prices : OrangePrices) (plan : PurchasePlan) : ℝ :=
  prices.cost_a * plan.kg_a + prices.cost_b * plan.kg_b

/-- Calculates the profit of a purchase plan --/
def profit (prices : OrangePrices) (plan : PurchasePlan) : ℝ :=
  (prices.sell_a - prices.cost_a) * plan.kg_a + (prices.sell_b - prices.cost_b) * plan.kg_b

/-- The main theorem to prove --/
theorem orange_profit_maximization (prices : OrangePrices) 
    (h1 : prices.sell_a = 16)
    (h2 : prices.sell_b = 24)
    (h3 : total_cost prices {kg_a := 15, kg_b := 20} = 430)
    (h4 : total_cost prices {kg_a := 10, kg_b := 8} = 212)
    (h5 : ∀ plan : PurchasePlan, plan.kg_a + plan.kg_b = 100 → 
      1160 ≤ total_cost prices plan ∧ total_cost prices plan ≤ 1168) :
  prices.cost_a = 10 ∧ 
  prices.cost_b = 14 ∧
  (∀ plan : PurchasePlan, plan.kg_a + plan.kg_b = 100 → 
    profit prices plan ≤ profit prices {kg_a := 58, kg_b := 42}) ∧
  profit prices {kg_a := 58, kg_b := 42} = 768 := by
  sorry


end orange_profit_maximization_l4009_400932


namespace phones_to_repair_per_person_l4009_400948

theorem phones_to_repair_per_person
  (initial_phones : ℕ)
  (repaired_phones : ℕ)
  (new_phones : ℕ)
  (h1 : initial_phones = 15)
  (h2 : repaired_phones = 3)
  (h3 : new_phones = 6)
  (h4 : repaired_phones ≤ initial_phones) :
  (initial_phones - repaired_phones + new_phones) / 2 = 9 := by
sorry

end phones_to_repair_per_person_l4009_400948


namespace outer_circle_radius_l4009_400982

theorem outer_circle_radius (r : ℝ) : 
  r > 0 ∧ 
  (π * (1.2 * r)^2 - π * 3^2) = (π * r^2 - π * 6^2) * 2.109375 → 
  r = 10 := by
  sorry

end outer_circle_radius_l4009_400982


namespace joshua_bottle_caps_l4009_400903

/-- The number of bottle caps Joshua bought -/
def bottle_caps_bought (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that the number of bottle caps Joshua bought
    is the difference between his final and initial counts -/
theorem joshua_bottle_caps 
  (initial : ℕ) 
  (final : ℕ) 
  (h1 : initial = 40) 
  (h2 : final = 47) :
  bottle_caps_bought initial final = 7 :=
by
  sorry

end joshua_bottle_caps_l4009_400903


namespace largest_of_five_consecutive_even_integers_l4009_400967

def sum_of_first_n_even_integers (n : ℕ) : ℕ := 2 * n * (n + 1)

def sum_of_five_consecutive_even_integers (largest : ℕ) : ℕ :=
  (largest - 8) + (largest - 6) + (largest - 4) + (largest - 2) + largest

theorem largest_of_five_consecutive_even_integers :
  ∃ (largest : ℕ), 
    sum_of_first_n_even_integers 15 = sum_of_five_consecutive_even_integers largest ∧
    largest = 52 := by
  sorry

end largest_of_five_consecutive_even_integers_l4009_400967


namespace f_max_min_l4009_400951

-- Define the function
def f (x : ℝ) : ℝ := |x^2 - x| + |x + 1|

-- State the theorem
theorem f_max_min :
  ∃ (max min : ℝ),
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x ≤ max) ∧
    (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = max) ∧
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → min ≤ f x) ∧
    (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = min) ∧
    max = 7 ∧ min = 1 :=
by sorry

end f_max_min_l4009_400951


namespace exponential_function_values_l4009_400949

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_values 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a 3 = 8) : 
  f a 4 = 16 ∧ f a (-4) = 1/16 := by
  sorry

end exponential_function_values_l4009_400949


namespace power_product_equality_l4009_400989

theorem power_product_equality (a : ℝ) : a * a^2 * (-a)^3 = -a^6 := by
  sorry

end power_product_equality_l4009_400989


namespace five_fold_f_of_one_l4009_400923

def f (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem five_fold_f_of_one : f (f (f (f (f 1)))) = 4687 := by
  sorry

end five_fold_f_of_one_l4009_400923


namespace y_value_proof_l4009_400968

theorem y_value_proof (y : ℝ) (h : 8 / y^3 = y / 32) : y = 4 := by
  sorry

end y_value_proof_l4009_400968


namespace smallest_solution_of_equation_l4009_400996

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x = 39/8 ∧
  x ≠ 0 ∧ x ≠ 3 ∧
  (3*x)/(x-3) + (3*x^2-27)/x = 14 ∧
  ∀ (y : ℝ), y ≠ 0 → y ≠ 3 → (3*y)/(y-3) + (3*y^2-27)/y = 14 → y ≥ x :=
by sorry

end smallest_solution_of_equation_l4009_400996


namespace sweater_markup_l4009_400944

theorem sweater_markup (wholesale : ℝ) (retail : ℝ) (h1 : retail > 0) (h2 : wholesale > 0) :
  (0.4 * retail = 1.35 * wholesale) →
  (retail - wholesale) / wholesale * 100 = 237.5 := by
sorry

end sweater_markup_l4009_400944


namespace curve_self_intersection_l4009_400912

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := 2 * t^2 - 4

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 6 * t^2 + 11 * t - 6

/-- The theorem stating that the curve intersects itself at (18, -44√11 - 6) -/
theorem curve_self_intersection :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    x t₁ = x t₂ ∧ 
    y t₁ = y t₂ ∧ 
    x t₁ = 18 ∧ 
    y t₁ = -44 * Real.sqrt 11 - 6 :=
sorry

end curve_self_intersection_l4009_400912


namespace factorization_1_factorization_2_factorization_3_l4009_400918

-- Define the condition for factoring quadratic trinomials
def is_factorizable (p q m n : ℤ) : Prop :=
  q = m * n ∧ p = m + n

-- Theorem 1
theorem factorization_1 : ∀ x : ℤ, x^2 - 7*x + 12 = (x - 3) * (x - 4) :=
  sorry

-- Theorem 2
theorem factorization_2 : ∀ x y : ℤ, (x - y)^2 + 4*(x - y) + 3 = (x - y + 1) * (x - y + 3) :=
  sorry

-- Theorem 3
theorem factorization_3 : ∀ a b : ℤ, (a + b) * (a + b - 2) - 3 = (a + b - 3) * (a + b + 1) :=
  sorry

end factorization_1_factorization_2_factorization_3_l4009_400918


namespace smallest_m_for_integral_solutions_l4009_400905

def has_integral_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℤ, a * x^2 + b * x + c = 0

theorem smallest_m_for_integral_solutions :
  (∀ m : ℤ, m > 0 ∧ m < 170 → ¬ has_integral_solutions 10 (-m) 720) ∧
  has_integral_solutions 10 (-170) 720 :=
sorry

end smallest_m_for_integral_solutions_l4009_400905


namespace expression_value_l4009_400935

theorem expression_value (x y z : ℚ) 
  (eq1 : 2 * x - y = 4)
  (eq2 : 3 * x + z = 7)
  (eq3 : y = 2 * z) :
  6 * x - 3 * y + 3 * z = 51 / 4 := by
  sorry

end expression_value_l4009_400935


namespace new_ratio_after_addition_l4009_400902

theorem new_ratio_after_addition : 
  ∀ (x y : ℤ), 
    x * 4 = y →  -- The two integers are in the ratio of 1 to 4
    y = 48 →     -- The larger integer is 48
    (x + 12) * 2 = y  -- The new ratio after adding 12 to the smaller integer is 1:2
    := by sorry

end new_ratio_after_addition_l4009_400902


namespace a_range_l4009_400928

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3/4| ≤ 1/4
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- Define the range of x for p
def p_range (x : ℝ) : Prop := 1/2 ≤ x ∧ x ≤ 1

-- Define the range of x for q
def q_range (x a : ℝ) : Prop := a ≤ x ∧ x ≤ a + 1

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p_range x → q_range x a) ∧
  ¬(∀ x, q_range x a → p_range x)

-- State the theorem
theorem a_range :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end a_range_l4009_400928


namespace cupcakes_remaining_l4009_400963

/-- The number of cupcake packages Maggi had -/
def packages : ℝ := 3.5

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 7

/-- The number of cupcakes Maggi ate -/
def eaten_cupcakes : ℝ := 5.75

/-- The number of cupcakes left after Maggi ate some -/
def cupcakes_left : ℝ := packages * cupcakes_per_package - eaten_cupcakes

theorem cupcakes_remaining : cupcakes_left = 18.75 := by
  sorry

end cupcakes_remaining_l4009_400963


namespace factorization_of_2m_squared_minus_18_l4009_400922

theorem factorization_of_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end factorization_of_2m_squared_minus_18_l4009_400922


namespace gain_percent_calculation_l4009_400956

/-- 
If the cost price of 50 articles is equal to the selling price of 15 articles, 
then the gain percent is 233.33%.
-/
theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 15 * S) : 
  (S - C) / C * 100 = 233.33 := by
  sorry

end gain_percent_calculation_l4009_400956


namespace married_men_fraction_l4009_400900

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : 0 < total_women) :
  let single_women := (3 * total_women) / 7
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  (single_women : ℚ) / total_women = 3 / 7 →
  (married_men : ℚ) / total_people = 4 / 11 := by
  sorry

end married_men_fraction_l4009_400900


namespace alberts_to_bettys_age_ratio_l4009_400958

/-- Proves that the ratio of Albert's age to Betty's age is 4:1 given the specified conditions -/
theorem alberts_to_bettys_age_ratio :
  ∀ (albert_age mary_age betty_age : ℕ),
    albert_age = 2 * mary_age →
    mary_age = albert_age - 14 →
    betty_age = 7 →
    (albert_age : ℚ) / betty_age = 4 / 1 := by
  sorry

end alberts_to_bettys_age_ratio_l4009_400958


namespace blood_expiration_date_l4009_400985

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The number of days in a non-leap year -/
def days_per_year : ℕ := 365

/-- The number of days in January -/
def days_in_january : ℕ := 31

/-- The number of days in February (non-leap year) -/
def days_in_february : ℕ := 28

/-- Calculate the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The expiration time of blood in seconds -/
def blood_expiration_time : ℕ := factorial 11

theorem blood_expiration_date :
  let total_days : ℕ := blood_expiration_time / seconds_per_day
  let days_in_second_year : ℕ := total_days - days_per_year
  let days_after_january : ℕ := days_in_second_year - days_in_january
  days_after_january = days_in_february + 8 :=
by sorry

end blood_expiration_date_l4009_400985


namespace largest_digit_for_two_digit_quotient_l4009_400934

theorem largest_digit_for_two_digit_quotient :
  ∀ n : ℕ, n ≤ 4 ∧ (n * 100 + 5) / 5 < 100 ∧
  ∀ m : ℕ, m > n → (m * 100 + 5) / 5 ≥ 100 →
  4 = n :=
sorry

end largest_digit_for_two_digit_quotient_l4009_400934


namespace max_product_of_three_l4009_400994

def S : Finset Int := {-9, -7, -3, 1, 4, 6}

theorem max_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  a * b * c ≤ 378 ∧ ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 378 :=
by sorry

end max_product_of_three_l4009_400994


namespace f_negative_a_eq_zero_l4009_400973

/-- Given a real-valued function f(x) = x³ + x + 1 and a real number a such that f(a) = 2,
    prove that f(-a) = 0. -/
theorem f_negative_a_eq_zero (a : ℝ) (h : a^3 + a + 1 = 2) :
  (-a)^3 + (-a) + 1 = 0 := by
  sorry

end f_negative_a_eq_zero_l4009_400973


namespace cat_stairs_ways_l4009_400920

def stair_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 1
  | m + 4 => stair_ways m + stair_ways (m + 1) + stair_ways (m + 2)

theorem cat_stairs_ways :
  stair_ways 10 = 12 :=
by sorry

end cat_stairs_ways_l4009_400920


namespace divisible_by_38_count_l4009_400915

def numbers : List Nat := [3624, 36024, 360924, 3609924, 36099924, 360999924, 3609999924]

theorem divisible_by_38_count :
  (numbers.filter (·.mod 38 = 0)).length = 6 := by
  sorry

end divisible_by_38_count_l4009_400915
