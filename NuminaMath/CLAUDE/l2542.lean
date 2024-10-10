import Mathlib

namespace initial_roses_count_l2542_254283

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 3

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 12

/-- The number of roses after adding flowers -/
def final_roses : ℕ := 11

/-- The number of orchids after adding flowers -/
def final_orchids : ℕ := 20

/-- The difference between orchids and roses after adding flowers -/
def orchid_rose_difference : ℕ := 9

theorem initial_roses_count :
  initial_roses = 3 ∧
  initial_orchids = 12 ∧
  final_roses = 11 ∧
  final_orchids = 20 ∧
  orchid_rose_difference = 9 ∧
  final_orchids - final_roses = orchid_rose_difference ∧
  final_orchids - initial_orchids = final_roses - initial_roses :=
by sorry

end initial_roses_count_l2542_254283


namespace circle_plus_four_two_l2542_254294

-- Define the operation ⊕
def circle_plus (a b : ℝ) : ℝ := 2 * a + 5 * b

-- Statement to prove
theorem circle_plus_four_two : circle_plus 4 2 = 18 := by
  sorry

end circle_plus_four_two_l2542_254294


namespace min_clients_theorem_exists_solution_with_101_min_clients_is_101_l2542_254263

/-- Represents a repunit number with n ones -/
def repunit (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- The property that needs to be satisfied for the group of clients -/
def satisfies_property (m k : ℕ) : Prop :=
  ∃ n : ℕ, n > k ∧ k > 1 ∧ repunit n = repunit k * m

/-- The main theorem stating the minimum number of clients -/
theorem min_clients_theorem :
  ∀ m : ℕ, m > 1 → (satisfies_property m 2) → m ≥ 101 :=
by sorry

/-- The existence theorem proving there is a solution with 101 clients -/
theorem exists_solution_with_101 :
  satisfies_property 101 2 :=
by sorry

/-- The final theorem proving 101 is the minimum number of clients -/
theorem min_clients_is_101 :
  ∀ m : ℕ, m > 1 → satisfies_property m 2 → m ≥ 101 ∧ satisfies_property 101 2 :=
by sorry

end min_clients_theorem_exists_solution_with_101_min_clients_is_101_l2542_254263


namespace simplify_fraction_l2542_254221

theorem simplify_fraction (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) :
  (1 - 1 / x) / ((1 - x^2) / x) = -1 / (1 + x) := by
  sorry

end simplify_fraction_l2542_254221


namespace square_side_length_l2542_254268

theorem square_side_length (k : ℝ) (s d : ℝ) (h1 : s > 0) (h2 : d > 0) (h3 : s + d = k) (h4 : d = s * Real.sqrt 2) :
  s = k / (1 + Real.sqrt 2) := by
sorry

end square_side_length_l2542_254268


namespace system_solution_form_l2542_254243

theorem system_solution_form (x y : ℝ) : 
  x + 2*y = 5 → 4*x*y = 9 → 
  ∃ (a b c d : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    ((x = (a + b * Real.sqrt c) / d) ∨ (x = (a - b * Real.sqrt c) / d)) ∧
    a = 5 ∧ b = 1 ∧ c = 7 ∧ d = 2 :=
by sorry

end system_solution_form_l2542_254243


namespace polynomial_equality_l2542_254231

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b = (x - 1)*(x + 4)) → a = 3 ∧ b = -4 := by
  sorry

end polynomial_equality_l2542_254231


namespace ellipse_y_axis_l2542_254296

/-- The equation represents an ellipse with focal points on the y-axis -/
theorem ellipse_y_axis (x y : ℝ) : 
  (x^2 / (Real.sin (Real.sqrt 2) - Real.sin (Real.sqrt 3))) + 
  (y^2 / (Real.cos (Real.sqrt 2) - Real.cos (Real.sqrt 3))) = 1 →
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

end ellipse_y_axis_l2542_254296


namespace smallest_third_side_of_right_triangle_l2542_254233

theorem smallest_third_side_of_right_triangle (a b c : ℝ) :
  a = 5 →
  b = 4 →
  c > 0 →
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 →
  3 ≤ c :=
by sorry

end smallest_third_side_of_right_triangle_l2542_254233


namespace largest_intersection_is_eight_l2542_254209

/-- A polynomial of degree 6 -/
def P (a b c : ℝ) (x : ℝ) : ℝ :=
  x^6 - 14*x^5 + 45*x^4 - 30*x^3 + a*x^2 + b*x + c

/-- A linear function -/
def L (d e : ℝ) (x : ℝ) : ℝ :=
  d*x + e

/-- The difference between P and L -/
def Q (a b c d e : ℝ) (x : ℝ) : ℝ :=
  P a b c x - L d e x

theorem largest_intersection_is_eight (a b c d e : ℝ) :
  (∃ p q r : ℝ, p < q ∧ q < r ∧
    ∀ x : ℝ, Q a b c d e x = 0 ↔ (x = p ∨ x = q ∨ x = r) ∧
    ∀ x : ℝ, x ≠ p ∧ x ≠ q ∧ x ≠ r → Q a b c d e x > 0) →
  r = 8 :=
sorry

end largest_intersection_is_eight_l2542_254209


namespace vacation_cost_problem_l2542_254250

/-- The vacation cost problem -/
theorem vacation_cost_problem 
  (alice_paid bob_paid carol_paid dave_paid : ℚ)
  (h_alice : alice_paid = 160)
  (h_bob : bob_paid = 120)
  (h_carol : carol_paid = 140)
  (h_dave : dave_paid = 200)
  (a b : ℚ) 
  (h_equal_split : (alice_paid + bob_paid + carol_paid + dave_paid) / 4 = 
                   alice_paid - a)
  (h_bob_contribution : (alice_paid + bob_paid + carol_paid + dave_paid) / 4 = 
                        bob_paid + b) :
  a - b = -35 := by
  sorry

end vacation_cost_problem_l2542_254250


namespace total_shaded_area_specific_total_shaded_area_l2542_254219

/-- The total shaded area of three overlapping rectangles -/
theorem total_shaded_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ)
  (shared_side triple_overlap_width : ℕ) : ℕ :=
  let rect1_area := rect1_width * rect1_height
  let rect2_area := rect2_width * rect2_height
  let rect3_area := rect3_width * rect3_height
  let overlap_area := shared_side * shared_side
  let triple_overlap_area := triple_overlap_width * shared_side
  rect1_area + rect2_area + rect3_area - overlap_area - triple_overlap_area

/-- The total shaded area of the specific configuration is 136 square units -/
theorem specific_total_shaded_area :
  total_shaded_area 4 15 5 10 3 18 4 3 = 136 := by
  sorry

end total_shaded_area_specific_total_shaded_area_l2542_254219


namespace sum_digits_ratio_bound_l2542_254207

/-- Sum of digits function -/
def S (n : ℕ+) : ℕ := sorry

/-- The theorem stating the upper bound and its achievability -/
theorem sum_digits_ratio_bound :
  (∀ n : ℕ+, (S n : ℚ) / (S (16 * n) : ℚ) ≤ 13) ∧
  (∃ n : ℕ+, (S n : ℚ) / (S (16 * n) : ℚ) = 13) :=
sorry

end sum_digits_ratio_bound_l2542_254207


namespace total_flight_distance_l2542_254239

/-- The total distance to fly from Germany to Russia and then return to Spain,
    given the distances between Spain-Russia and Spain-Germany. -/
theorem total_flight_distance (spain_russia spain_germany : ℕ) 
  (h1 : spain_russia = 7019)
  (h2 : spain_germany = 1615) :
  spain_russia + (spain_russia - spain_germany) = 12423 :=
by sorry

end total_flight_distance_l2542_254239


namespace f_max_value_l2542_254271

/-- The quadratic function f(x) = -x^2 + 2x + 4 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 4

/-- The maximum value of f(x) is 5 -/
theorem f_max_value : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M :=
  sorry

end f_max_value_l2542_254271


namespace quadratic_exponent_condition_l2542_254212

theorem quadratic_exponent_condition (a : ℝ) : 
  (∀ x, ∃ p q r : ℝ, x^(a^2 - 7) - 3*x - 2 = p*x^2 + q*x + r) → 
  (a = 3 ∨ a = -3) := by
  sorry

end quadratic_exponent_condition_l2542_254212


namespace geometric_sequence_property_l2542_254273

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that for a geometric sequence where a_1 * a_5 = a_3, the value of a_3 is 1. -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : IsGeometricSequence a) 
  (h_prop : a 1 * a 5 = a 3) : 
  a 3 = 1 := by
  sorry

end geometric_sequence_property_l2542_254273


namespace arrangement_count_l2542_254211

/-- The number of people in the row -/
def n : ℕ := 5

/-- The number of arrangements where A and B are adjacent -/
def adjacent_arrangements : ℕ := 48

/-- The total number of arrangements of n people -/
def total_arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of arrangements where A and B are not adjacent -/
def non_adjacent_arrangements (n : ℕ) : ℕ := total_arrangements n - adjacent_arrangements

/-- The number of arrangements where A and B are not adjacent and A is to the left of B -/
def target_arrangements (n : ℕ) : ℕ := non_adjacent_arrangements n / 2

theorem arrangement_count : target_arrangements n = 36 := by
  sorry

end arrangement_count_l2542_254211


namespace first_discount_percentage_l2542_254236

theorem first_discount_percentage (original_price final_price : ℝ) (second_discount : ℝ) :
  original_price = 26.67 →
  final_price = 15 →
  second_discount = 25 →
  ∃ first_discount : ℝ,
    first_discount = 25 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by
  sorry

end first_discount_percentage_l2542_254236


namespace negative_sum_positive_product_l2542_254203

theorem negative_sum_positive_product (a b : ℝ) : 
  a + b < 0 → ab > 0 → a < 0 ∧ b < 0 := by sorry

end negative_sum_positive_product_l2542_254203


namespace share_ratio_problem_l2542_254278

theorem share_ratio_problem (total : ℝ) (share_A : ℝ) (ratio_B_C : ℚ) 
  (h_total : total = 116000)
  (h_share_A : share_A = 29491.525423728814)
  (h_ratio_B_C : ratio_B_C = 5/6) :
  ∃ (share_B : ℝ), share_A / share_B = 3/4 := by
  sorry

end share_ratio_problem_l2542_254278


namespace square_perimeter_l2542_254259

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 324 → 
  area = side * side →
  perimeter = 4 * side →
  perimeter = 72 := by
  sorry

end square_perimeter_l2542_254259


namespace complex_equation_sum_l2542_254299

theorem complex_equation_sum (a b : ℝ) : (1 + 2*I)*I = a + b*I → a + b = -1 := by
  sorry

end complex_equation_sum_l2542_254299


namespace sum_of_digits_is_400_l2542_254241

/-- A number system with base r -/
structure BaseR where
  r : ℕ
  h_r : r ≤ 400

/-- A number x in base r of the form ppqq -/
structure NumberX (b : BaseR) where
  p : ℕ
  q : ℕ
  h_pq : 7 * q = 17 * p
  x : ℕ
  h_x : x = p * b.r^3 + p * b.r^2 + q * b.r + q

/-- The square of x is a seven-digit palindrome with middle digit zero -/
def is_palindrome_square (b : BaseR) (x : NumberX b) : Prop :=
  ∃ (a c : ℕ),
    x.x^2 = a * b.r^6 + c * b.r^5 + c * b.r^4 + 0 * b.r^3 + c * b.r^2 + c * b.r + a

/-- The sum of digits of x^2 in base r -/
def sum_of_digits (b : BaseR) (x : NumberX b) : ℕ :=
  sorry  -- Definition of sum of digits

/-- Main theorem -/
theorem sum_of_digits_is_400 (b : BaseR) (x : NumberX b) 
    (h_palindrome : is_palindrome_square b x) : 
    sum_of_digits b x = 400 := by
  sorry

end sum_of_digits_is_400_l2542_254241


namespace factor_implies_s_value_l2542_254242

theorem factor_implies_s_value (m s : ℝ) : 
  (m - 8) ∣ (m^2 - s*m - 24) → s = 5 := by
sorry

end factor_implies_s_value_l2542_254242


namespace length_of_AB_l2542_254220

-- Define the triangle
def Triangle (A B C : ℝ) := True

-- Define the right angle
def is_right_angle (B : ℝ) := B = 90

-- Define the angle A
def angle_A (A : ℝ) := A = 40

-- Define the length of side BC
def side_BC (BC : ℝ) := BC = 7

-- Theorem statement
theorem length_of_AB (A B C BC : ℝ) 
  (triangle : Triangle A B C) 
  (right_angle : is_right_angle B) 
  (angle_a : angle_A A) 
  (side_bc : side_BC BC) : 
  ∃ (AB : ℝ), abs (AB - 8.3) < 0.1 := by
  sorry

end length_of_AB_l2542_254220


namespace molly_total_distance_l2542_254229

/-- The total distance Molly swam over two days -/
def total_distance (saturday_distance sunday_distance : ℕ) : ℕ :=
  saturday_distance + sunday_distance

/-- Theorem stating that Molly's total swimming distance is 430 meters -/
theorem molly_total_distance : total_distance 250 180 = 430 := by
  sorry

end molly_total_distance_l2542_254229


namespace ellipse_equation_l2542_254269

/-- The standard equation of an ellipse with given foci and major axis length -/
theorem ellipse_equation (a b c : ℝ) (h1 : c^2 = 5) (h2 : a = 5) (h3 : b^2 = a^2 - c^2) :
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 25) + (y^2 / 20) = 1 :=
by sorry

end ellipse_equation_l2542_254269


namespace point_B_coordinates_l2542_254235

def point := ℝ × ℝ

def vector := ℝ × ℝ

def point_A : point := (-1, 5)

def vector_AB : vector := (6, 9)

def point_B : point := (5, 14)

def vector_between (p q : point) : vector :=
  (q.1 - p.1, q.2 - p.2)

theorem point_B_coordinates :
  vector_between point_A point_B = vector_AB :=
by sorry

end point_B_coordinates_l2542_254235


namespace brand_a_households_l2542_254226

theorem brand_a_households (total : ℕ) (neither : ℕ) (both : ℕ) (ratio : ℕ) :
  total = 160 →
  neither = 80 →
  both = 5 →
  ratio = 3 →
  ∃ (only_a only_b : ℕ),
    total = neither + only_a + only_b + both ∧
    only_b = ratio * both ∧
    only_a = 60 :=
by sorry

end brand_a_households_l2542_254226


namespace floor_sqrt_225_l2542_254292

theorem floor_sqrt_225 : ⌊Real.sqrt 225⌋ = 15 := by sorry

end floor_sqrt_225_l2542_254292


namespace clarinet_fraction_in_band_l2542_254261

theorem clarinet_fraction_in_band (total_band : ℕ) (flutes_in : ℕ) (trumpets_in : ℕ) (pianists_in : ℕ) (clarinets_total : ℕ) :
  total_band = 53 →
  flutes_in = 16 →
  trumpets_in = 20 →
  pianists_in = 2 →
  clarinets_total = 30 →
  (total_band - (flutes_in + trumpets_in + pianists_in)) / clarinets_total = 1 / 2 :=
by
  sorry

#check clarinet_fraction_in_band

end clarinet_fraction_in_band_l2542_254261


namespace farm_section_area_l2542_254275

/-- Given a farm with a total area of 300 acres divided into 5 equal sections,
    prove that the area of each section is 60 acres. -/
theorem farm_section_area (total_area : ℝ) (num_sections : ℕ) (section_area : ℝ) :
  total_area = 300 ∧ num_sections = 5 ∧ section_area * num_sections = total_area →
  section_area = 60 := by
  sorry

end farm_section_area_l2542_254275


namespace single_elimination_tournament_matches_l2542_254280

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  initial_players : ℕ
  matches_played : ℕ

/-- The number of matches needed to determine a champion in a single-elimination tournament -/
def matches_needed (tournament : SingleEliminationTournament) : ℕ :=
  tournament.initial_players - 1

theorem single_elimination_tournament_matches 
  (tournament : SingleEliminationTournament)
  (h : tournament.initial_players = 512) :
  matches_needed tournament = 511 := by
  sorry

end single_elimination_tournament_matches_l2542_254280


namespace prob_two_green_apples_l2542_254282

/-- The probability of selecting 2 green apples from a set of 7 apples, where 3 are green -/
theorem prob_two_green_apples (total : ℕ) (green : ℕ) (choose : ℕ) 
  (h_total : total = 7) 
  (h_green : green = 3) 
  (h_choose : choose = 2) :
  (Nat.choose green choose : ℚ) / (Nat.choose total choose : ℚ) = 1 / 7 := by
  sorry

#check prob_two_green_apples

end prob_two_green_apples_l2542_254282


namespace zeros_of_f_l2542_254287

def f (x : ℝ) : ℝ := x^2 - x - 2

theorem zeros_of_f :
  {x : ℝ | f x = 0} = {-1, 2} := by sorry

end zeros_of_f_l2542_254287


namespace subtraction_multiplication_equality_l2542_254252

theorem subtraction_multiplication_equality : 10111 - 10 * 2 * 5 = 10011 := by
  sorry

end subtraction_multiplication_equality_l2542_254252


namespace intersection_points_l2542_254265

-- Define g as a function from real numbers to real numbers
variable (g : ℝ → ℝ)

-- Define the property that g is invertible
def IsInvertible (g : ℝ → ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x, h (g x) = x) ∧ (∀ y, g (h y) = y)

-- Theorem statement
theorem intersection_points (h : IsInvertible g) :
  (∃! n : Nat, ∃ s : Finset ℝ, s.card = n ∧
    (∀ x : ℝ, x ∈ s ↔ g (x^3) = g (x^6))) ∧
  (∃ s : Finset ℝ, s.card = 3 ∧
    (∀ x : ℝ, x ∈ s ↔ g (x^3) = g (x^6))) :=
sorry

end intersection_points_l2542_254265


namespace min_convex_division_rotated_ngon_l2542_254295

/-- A regular n-gon. -/
structure RegularNGon (n : ℕ) where
  -- Add necessary fields here

/-- Rotate a regular n-gon by an angle around its center. -/
def rotate (M : RegularNGon n) (angle : ℝ) : RegularNGon n :=
  sorry

/-- The union of two regular n-gons. -/
def union (M M' : RegularNGon n) : Set (ℝ × ℝ) :=
  sorry

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields here

/-- The minimum number of convex polygons needed to divide a set. -/
def minConvexDivision (S : Set (ℝ × ℝ)) : ℕ :=
  sorry

theorem min_convex_division_rotated_ngon (n : ℕ) (M : RegularNGon n) :
  minConvexDivision (union M (rotate M (π / n))) = n + 1 :=
sorry

end min_convex_division_rotated_ngon_l2542_254295


namespace simplify_sum_of_radicals_l2542_254270

theorem simplify_sum_of_radicals : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 2 := by
sorry

end simplify_sum_of_radicals_l2542_254270


namespace smallest_addend_to_palindrome_l2542_254205

/-- A function that checks if a positive integer is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The smallest positive integer that can be added to 2002 to produce a larger palindrome -/
def smallestAddend : ℕ := 110

theorem smallest_addend_to_palindrome : 
  (isPalindrome 2002) ∧ 
  (isPalindrome (2002 + smallestAddend)) ∧ 
  (∀ k : ℕ, k < smallestAddend → ¬ isPalindrome (2002 + k)) := by sorry

end smallest_addend_to_palindrome_l2542_254205


namespace coronavirus_cases_difference_l2542_254202

theorem coronavirus_cases_difference (new_york california texas : ℕ) : 
  new_york = 2000 →
  california = new_york / 2 →
  new_york + california + texas = 3600 →
  texas < california →
  california - texas = 400 :=
by sorry

end coronavirus_cases_difference_l2542_254202


namespace critical_point_theorem_l2542_254237

def sequence_property (x : ℕ → ℝ) : Prop :=
  (∀ n, x n > 0) ∧
  (8 * x 2 - 7 * x 1) * (x 1)^7 = 8 ∧
  (∀ k ≥ 2, (x (k+1)) * (x (k-1)) - (x k)^2 = ((x (k-1))^8 - (x k)^8) / ((x k)^7 * (x (k-1))^7))

def monotonically_decreasing (x : ℕ → ℝ) : Prop :=
  ∀ n, x (n+1) ≤ x n

def not_monotonic (x : ℕ → ℝ) : Prop :=
  ∃ m n, m < n ∧ x m < x n

theorem critical_point_theorem (x : ℕ → ℝ) (h : sequence_property x) :
  ∃ a : ℝ, a = 8^(1/8) ∧
    ((x 1 > a → monotonically_decreasing x) ∧
     (0 < x 1 ∧ x 1 < a → not_monotonic x)) :=
sorry

end critical_point_theorem_l2542_254237


namespace not_all_squares_congruent_l2542_254222

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem: It is false that all squares are congruent to each other
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

-- Other properties of squares (for completeness, not used in the proof)
def convex (s : Square) : Prop := true
def equiangular (s : Square) : Prop := true
def regular_polygon (s : Square) : Prop := true
def similar (s1 s2 : Square) : Prop := true

end not_all_squares_congruent_l2542_254222


namespace no_constant_absolute_value_inequality_l2542_254285

theorem no_constant_absolute_value_inequality :
  ¬ ∃ (a b c : ℝ), ∀ (x y : ℝ), 
    |x + a| + |x + y + b| + |y + c| > |x| + |x + y| + |y| := by
  sorry

end no_constant_absolute_value_inequality_l2542_254285


namespace black_grid_probability_l2542_254216

/-- Represents a 4x4 grid of squares --/
def Grid := Fin 4 → Fin 4 → Bool

/-- Rotates the grid 90 degrees clockwise --/
def rotate (g : Grid) : Grid := sorry

/-- Applies the painting rule: white squares adjacent to black become black --/
def applyPaintRule (g : Grid) : Grid := sorry

/-- Checks if the entire grid is black --/
def allBlack (g : Grid) : Prop := ∀ i j, g i j = true

/-- Generates a random initial grid --/
def randomGrid : Grid := sorry

/-- The probability of a grid being entirely black after operations --/
def blackProbability : ℝ := sorry

theorem black_grid_probability :
  ∃ (p : ℝ), 0 < p ∧ p < 1 ∧ blackProbability = p := by sorry

end black_grid_probability_l2542_254216


namespace loan_balance_after_ten_months_l2542_254215

/-- Represents a loan with monthly payments -/
structure Loan where
  monthly_payment : ℕ
  total_months : ℕ
  current_balance : ℕ

/-- Calculates the remaining balance of a loan after a given number of months -/
def remaining_balance (loan : Loan) (months : ℕ) : ℕ :=
  loan.current_balance - loan.monthly_payment * months

/-- Theorem: Given a loan where $10 is paid back monthly, and half of the loan has been repaid 
    after 6 months, the remaining balance after 10 months will be $20 -/
theorem loan_balance_after_ten_months 
  (loan : Loan)
  (h1 : loan.monthly_payment = 10)
  (h2 : loan.total_months = 6)
  (h3 : loan.current_balance = loan.monthly_payment * loan.total_months) :
  remaining_balance loan 4 = 20 := by
  sorry


end loan_balance_after_ten_months_l2542_254215


namespace vector_operation_proof_l2542_254224

theorem vector_operation_proof :
  (4 : ℝ) • (![3, -5] : Fin 2 → ℝ) - (3 : ℝ) • (![2, -6] : Fin 2 → ℝ) = ![6, -2] := by
  sorry

end vector_operation_proof_l2542_254224


namespace power_function_problem_l2542_254251

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- Define the problem statement
theorem power_function_problem (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 3 = Real.sqrt 3) : 
  f 9 = 3 := by
  sorry

end power_function_problem_l2542_254251


namespace y_value_proof_l2542_254276

theorem y_value_proof : ∀ y : ℚ, (1/4 - 1/5 = 4/y) → y = 80 := by
  sorry

end y_value_proof_l2542_254276


namespace largest_circle_equation_l2542_254264

/-- The line equation ax - y - 4a - 2 = 0, where a is a real number -/
def line_equation (a x y : ℝ) : Prop := a * x - y - 4 * a - 2 = 0

/-- The center of the circle is at point (2, 0) -/
def circle_center : ℝ × ℝ := (2, 0)

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

theorem largest_circle_equation :
  ∃ (r : ℝ), r > 0 ∧
  (∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ circle_equation x y circle_center.1 circle_center.2 r) ∧
  (∀ r' : ℝ, r' > 0 →
    (∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ circle_equation x y circle_center.1 circle_center.2 r') →
    r' ≤ r) ∧
  (∀ x y : ℝ, circle_equation x y circle_center.1 circle_center.2 r ↔ (x - 2)^2 + y^2 = 8) :=
sorry

end largest_circle_equation_l2542_254264


namespace volumeAsFractionOfLitre_l2542_254255

-- Define the conversion factor from litres to millilitres
def litreToMl : ℝ := 1000

-- Define the volume in millilitres
def volumeMl : ℝ := 30

-- Theorem to prove
theorem volumeAsFractionOfLitre : (volumeMl / litreToMl) = 0.03 := by
  sorry

end volumeAsFractionOfLitre_l2542_254255


namespace guitar_difference_is_three_l2542_254277

/-- The number of fewer 8 string guitars compared to normal guitars -/
def guitar_difference : ℕ :=
  let num_basses : ℕ := 3
  let strings_per_bass : ℕ := 4
  let num_normal_guitars : ℕ := 2 * num_basses
  let strings_per_normal_guitar : ℕ := 6
  let strings_per_8string_guitar : ℕ := 8
  let total_strings : ℕ := 72
  let normal_guitar_strings : ℕ := num_normal_guitars * strings_per_normal_guitar
  let bass_strings : ℕ := num_basses * strings_per_bass
  let remaining_strings : ℕ := total_strings - (normal_guitar_strings + bass_strings)
  let num_8string_guitars : ℕ := remaining_strings / strings_per_8string_guitar
  num_normal_guitars - num_8string_guitars

theorem guitar_difference_is_three :
  guitar_difference = 3 := by sorry

end guitar_difference_is_three_l2542_254277


namespace sum_f_two_and_neg_two_l2542_254217

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + a)^3

-- State the theorem
theorem sum_f_two_and_neg_two (a : ℝ) : 
  (∀ x : ℝ, f a (1 + x) = -f a (1 - x)) → f a 2 + f a (-2) = -26 :=
by
  sorry

end sum_f_two_and_neg_two_l2542_254217


namespace round_trip_distance_l2542_254234

/-- Calculates the distance of a round trip given upstream speed, downstream speed, and total time -/
theorem round_trip_distance 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : upstream_speed > 0)
  (h2 : downstream_speed > 0)
  (h3 : total_time > 0)
  (h4 : upstream_speed = 3)
  (h5 : downstream_speed = 9)
  (h6 : total_time = 8) :
  (let distance := (upstream_speed * downstream_speed * total_time) / (upstream_speed + downstream_speed)
   distance = 18) := by
  sorry

end round_trip_distance_l2542_254234


namespace tan_sum_specific_l2542_254206

theorem tan_sum_specific (a b : Real) 
  (ha : Real.tan a = 1/2) (hb : Real.tan b = 1/3) : 
  Real.tan (a + b) = 1 := by sorry

end tan_sum_specific_l2542_254206


namespace blocks_used_for_tower_and_house_l2542_254293

theorem blocks_used_for_tower_and_house : 
  let total_blocks : ℕ := 58
  let tower_blocks : ℕ := 27
  let house_blocks : ℕ := 53
  tower_blocks + house_blocks = 80 :=
by sorry

end blocks_used_for_tower_and_house_l2542_254293


namespace inequality_system_solutions_l2542_254208

def has_exactly_two_integer_solutions (m : ℝ) : Prop :=
  ∃! (x y : ℤ), (x < 1 ∧ x > m - 1) ∧ (y < 1 ∧ y > m - 1) ∧ x ≠ y

theorem inequality_system_solutions (m : ℝ) :
  has_exactly_two_integer_solutions m ↔ -1 ≤ m ∧ m < 0 :=
sorry

end inequality_system_solutions_l2542_254208


namespace lemon_heads_distribution_l2542_254284

def small_package : ℕ := 6
def medium_package : ℕ := 15
def large_package : ℕ := 30

def louis_small_packages : ℕ := 5
def louis_medium_packages : ℕ := 3
def louis_large_packages : ℕ := 2

def louis_eaten : ℕ := 54
def num_friends : ℕ := 4

theorem lemon_heads_distribution :
  let total := louis_small_packages * small_package + 
               louis_medium_packages * medium_package + 
               louis_large_packages * large_package
  let remaining := total - louis_eaten
  let per_friend := remaining / num_friends
  per_friend = 3 * small_package + 2 ∧ 
  remaining % num_friends = 1 := by sorry

end lemon_heads_distribution_l2542_254284


namespace fifth_reading_calculation_l2542_254210

theorem fifth_reading_calculation (r1 r2 r3 r4 : ℝ) (mean : ℝ) (h1 : r1 = 2) (h2 : r2 = 2.1) (h3 : r3 = 2) (h4 : r4 = 2.2) (h_mean : mean = 2) :
  ∃ r5 : ℝ, (r1 + r2 + r3 + r4 + r5) / 5 = mean ∧ r5 = 1.7 :=
by sorry

end fifth_reading_calculation_l2542_254210


namespace age_difference_l2542_254254

theorem age_difference (A B : ℕ) : B = 48 → A + 10 = 2 * (B - 10) → A - B = 18 := by
  sorry

end age_difference_l2542_254254


namespace triangle_sides_not_proportional_l2542_254218

theorem triangle_sides_not_proportional (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ¬∃ (m : ℝ), m > 0 ∧ a = m^a ∧ b = m^b ∧ c = m^c :=
sorry

end triangle_sides_not_proportional_l2542_254218


namespace multiplicative_inverse_300_mod_2399_l2542_254238

theorem multiplicative_inverse_300_mod_2399 :
  (39 : ℤ)^2 + 80^2 = 89^2 →
  (300 * 1832) % 2399 = 1 :=
by sorry

end multiplicative_inverse_300_mod_2399_l2542_254238


namespace union_of_A_and_B_l2542_254227

def A : Set ℝ := {x : ℝ | x^2 - x - 2 = 0}

def B : Set ℝ := {y : ℝ | ∃ x ∈ A, y = x + 3}

theorem union_of_A_and_B : A ∪ B = {-1, 2, 5} := by sorry

end union_of_A_and_B_l2542_254227


namespace time_conversion_l2542_254240

-- Define the conversion rates
def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60

-- Define the given time
def hours : ℕ := 3
def minutes : ℕ := 25

-- Theorem to prove
theorem time_conversion :
  (hours * minutes_per_hour + minutes) * seconds_per_minute = 12300 := by
  sorry

end time_conversion_l2542_254240


namespace largest_multiple_of_15_less_than_500_l2542_254288

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ m : ℕ, m * 15 < 500 → m * 15 ≤ 495 := by
sorry

end largest_multiple_of_15_less_than_500_l2542_254288


namespace polygon_sides_l2542_254274

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → ∃ n : ℕ, n = 9 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end polygon_sides_l2542_254274


namespace symmetric_complex_sum_l2542_254291

theorem symmetric_complex_sum (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  let w : ℂ := Complex.I * (Complex.I - 2)
  (z.re = w.re ∧ z.im = -w.im) → a + b = 1 := by
  sorry

end symmetric_complex_sum_l2542_254291


namespace recycling_team_points_l2542_254244

/-- Represents the recycling data for a team member -/
structure RecyclingData where
  paper : Nat
  plastic : Nat
  aluminum : Nat

/-- Calculates the points earned for a given recycling data -/
def calculate_points (data : RecyclingData) : Nat :=
  (data.paper / 12) + (data.plastic / 6) + (data.aluminum / 4)

/-- The recycling data for each team member -/
def team_data : List RecyclingData := [
  { paper := 35, plastic := 15, aluminum := 5 },   -- Zoe
  { paper := 28, plastic := 18, aluminum := 8 },   -- Friend 1
  { paper := 22, plastic := 10, aluminum := 6 },   -- Friend 2
  { paper := 40, plastic := 20, aluminum := 10 },  -- Friend 3
  { paper := 18, plastic := 12, aluminum := 8 }    -- Friend 4
]

/-- Theorem: The recycling team earned 28 points -/
theorem recycling_team_points : 
  (team_data.map calculate_points).sum = 28 := by
  sorry

end recycling_team_points_l2542_254244


namespace two_crayons_per_color_per_box_l2542_254249

/-- Represents a crayon factory with given production parameters -/
structure CrayonFactory where
  colors : ℕ
  boxes_per_hour : ℕ
  total_crayons : ℕ
  total_hours : ℕ

/-- Calculates the number of crayons of each color per box -/
def crayons_per_color_per_box (factory : CrayonFactory) : ℕ :=
  factory.total_crayons / (factory.boxes_per_hour * factory.total_hours * factory.colors)

/-- Theorem stating that for the given factory parameters, there are 2 crayons of each color per box -/
theorem two_crayons_per_color_per_box (factory : CrayonFactory) 
  (h1 : factory.colors = 4)
  (h2 : factory.boxes_per_hour = 5)
  (h3 : factory.total_crayons = 160)
  (h4 : factory.total_hours = 4) :
  crayons_per_color_per_box factory = 2 := by
  sorry

end two_crayons_per_color_per_box_l2542_254249


namespace quadratic_root_relation_l2542_254248

theorem quadratic_root_relation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + m*x₁ + 5 = 0 ∧ 
                x₂^2 + m*x₂ + 5 = 0 ∧ 
                x₁ = 2*|x₂| - 3) → 
  m = -9/2 := by
sorry

end quadratic_root_relation_l2542_254248


namespace right_triangle_existence_l2542_254286

theorem right_triangle_existence (a q : ℝ) (ha : a > 0) (hq : q > 0) :
  ∃ (b c : ℝ), 
    b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    (b^2 / c) = q :=
by sorry

end right_triangle_existence_l2542_254286


namespace f_properties_l2542_254258

def f (x : ℝ) := x^2

theorem f_properties : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
sorry

end f_properties_l2542_254258


namespace profit_percentage_l2542_254262

theorem profit_percentage (C S : ℝ) (h : C > 0) : 
  20 * C = 16 * S → (S - C) / C * 100 = 25 := by
  sorry

end profit_percentage_l2542_254262


namespace arithmetic_mean_geq_geometric_mean_l2542_254232

theorem arithmetic_mean_geq_geometric_mean (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end arithmetic_mean_geq_geometric_mean_l2542_254232


namespace negation_of_universal_statement_l2542_254297

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end negation_of_universal_statement_l2542_254297


namespace fraction_five_seventeenths_repetend_l2542_254223

/-- The repetend of a rational number in its decimal representation -/
def repetend (n d : ℕ) : List ℕ := sorry

/-- The length of the repetend of a rational number in its decimal representation -/
def repetendLength (n d : ℕ) : ℕ := sorry

theorem fraction_five_seventeenths_repetend :
  repetend 5 17 = [2, 9, 4, 1, 1, 7, 6] ∧ repetendLength 5 17 = 7 := by
  sorry

end fraction_five_seventeenths_repetend_l2542_254223


namespace unique_solution_is_one_l2542_254246

theorem unique_solution_is_one (n : ℕ) (hn : n ≥ 1) :
  (∃ (a b : ℕ), 
    (∀ (p : ℕ), Prime p → ¬(p^3 ∣ (a^2 + b + 3))) ∧
    ((a * b + 3 * b + 8) : ℚ) / (a^2 + b + 3 : ℚ) = n) 
  ↔ n = 1 := by
  sorry

end unique_solution_is_one_l2542_254246


namespace power_equality_l2542_254225

theorem power_equality (m : ℕ) : 5^m = 5 * 25^5 * 125^3 → m = 20 := by
  sorry

end power_equality_l2542_254225


namespace quadratic_sequence_bound_l2542_254260

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the solutions of a quadratic equation -/
structure QuadraticSolution where
  x₁ : ℝ
  x₂ : ℝ

/-- Function to get the next quadratic equation in the sequence -/
def nextEquation (eq : QuadraticEquation) (sol : QuadraticSolution) : QuadraticEquation :=
  { a := 1, b := -sol.x₁, c := -sol.x₂ }

/-- Theorem stating that the sequence of quadratic equations has at most 5 elements -/
theorem quadratic_sequence_bound
  (a₁ b₁ : ℝ)
  (h₁ : a₁ ≠ 0)
  (h₂ : b₁ ≠ 0)
  (initial : QuadraticEquation)
  (h₃ : initial = { a := 1, b := a₁, c := b₁ })
  (next : QuadraticEquation → QuadraticSolution → QuadraticEquation)
  (h₄ : ∀ eq sol, next eq sol = nextEquation eq sol) :
  ∃ n : ℕ, n ≤ 5 ∧ ∀ m : ℕ, m > n →
    ¬∃ (seq : ℕ → QuadraticEquation) (sols : ℕ → QuadraticSolution),
      (seq 0 = initial) ∧
      (∀ k < m, seq (k + 1) = next (seq k) (sols k)) ∧
      (∀ k < m, (sols k).x₁ ≤ (sols k).x₂) :=
sorry

end quadratic_sequence_bound_l2542_254260


namespace chord_length_l2542_254228

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end chord_length_l2542_254228


namespace perpendicular_vectors_x_value_l2542_254247

/-- Given two vectors a and b in ℝ², where a = (1, 2) and b = (x, -2),
    if a and b are perpendicular, then x = 4. -/
theorem perpendicular_vectors_x_value :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, -2]
  (∀ i, i < 2 → a i * b i = 0) →
  x = 4 := by
sorry

end perpendicular_vectors_x_value_l2542_254247


namespace tree_space_calculation_l2542_254200

/-- Given a road of length 166 feet where 16 trees are planted with 10 feet between each tree,
    prove that each tree occupies 1 square foot of sidewalk space. -/
theorem tree_space_calculation (road_length : ℝ) (num_trees : ℕ) (space_between : ℝ) : 
  road_length = 166 ∧ num_trees = 16 ∧ space_between = 10 → 
  (road_length - space_between * (num_trees - 1)) / num_trees = 1 :=
by sorry

end tree_space_calculation_l2542_254200


namespace original_denominator_problem_l2542_254266

theorem original_denominator_problem (d : ℕ) : 
  (3 : ℚ) / d ≠ 0 → 
  (6 : ℚ) / (d + 3) = 1 / 3 → 
  d = 15 := by
sorry

end original_denominator_problem_l2542_254266


namespace side_length_of_five_cubes_l2542_254257

/-- Given five equal cubes placed adjacent to each other forming a new solid with volume 625 cm³,
    prove that the side length of each cube is 5 cm. -/
theorem side_length_of_five_cubes (n : ℕ) (v : ℝ) (s : ℝ) : 
  n = 5 → v = 625 → v = n * s^3 → s = 5 := by sorry

end side_length_of_five_cubes_l2542_254257


namespace second_expression_value_l2542_254279

/-- Given that the average of (2a + 16) and x is 79, and a = 30, prove that x = 82 -/
theorem second_expression_value (a x : ℝ) : 
  ((2 * a + 16) + x) / 2 = 79 → a = 30 → x = 82 := by
  sorry

end second_expression_value_l2542_254279


namespace visitors_to_both_countries_l2542_254290

theorem visitors_to_both_countries 
  (total : ℕ) 
  (iceland : ℕ) 
  (norway : ℕ) 
  (neither : ℕ) 
  (h1 : total = 60) 
  (h2 : iceland = 35) 
  (h3 : norway = 23) 
  (h4 : neither = 33) : 
  ∃ (both : ℕ), both = 31 ∧ 
    total = iceland + norway - both + neither :=
by sorry

end visitors_to_both_countries_l2542_254290


namespace no_return_after_12_jumps_all_return_after_13_jumps_l2542_254281

/-- Represents a point on a circle -/
structure CirclePoint where
  position : ℕ

/-- The number of points on the circle -/
def n : ℕ := 12

/-- The jump function that moves a point to the next clockwise midpoint -/
def jump (p : CirclePoint) : CirclePoint :=
  ⟨(p.position + 1) % n⟩

/-- Applies the jump function k times -/
def jumpK (p : CirclePoint) (k : ℕ) : CirclePoint :=
  match k with
  | 0 => p
  | k + 1 => jump (jumpK p k)

theorem no_return_after_12_jumps :
  ∀ p : CirclePoint, jumpK p 12 ≠ p :=
sorry

theorem all_return_after_13_jumps :
  ∀ p : CirclePoint, jumpK p 13 = p :=
sorry

end no_return_after_12_jumps_all_return_after_13_jumps_l2542_254281


namespace sports_books_count_l2542_254201

theorem sports_books_count (total_books : ℕ) (school_books : ℕ) (sports_books : ℕ) 
  (h1 : total_books = 344)
  (h2 : school_books = 136)
  (h3 : total_books = school_books + sports_books) :
  sports_books = 208 := by
sorry

end sports_books_count_l2542_254201


namespace number_of_B_l2542_254267

/-- Given that the number of A is x and the number of B is a less than half of A,
    prove that the number of B is equal to (1/2)x - a. -/
theorem number_of_B (x a : ℝ) (hA : x ≥ 0) (hB : x ≥ 2 * a) :
  (1/2 : ℝ) * x - a = (1/2 : ℝ) * x - a :=
by sorry

end number_of_B_l2542_254267


namespace least_addition_for_divisibility_l2542_254253

theorem least_addition_for_divisibility : 
  (∃ (n : ℕ), 25 ∣ (1019 + n) ∧ ∀ (m : ℕ), m < n → ¬(25 ∣ (1019 + m))) ∧ 
  (∃ (n : ℕ), n = 6 ∧ 25 ∣ (1019 + n) ∧ ∀ (m : ℕ), m < n → ¬(25 ∣ (1019 + m))) :=
by sorry

end least_addition_for_divisibility_l2542_254253


namespace translation_product_l2542_254298

/-- Given a point P(-3, y) translated 3 units down and 2 units left to obtain point Q(x, -1), 
    the product xy equals -10. -/
theorem translation_product (y : ℝ) : 
  let x : ℝ := -3 - 2
  let y' : ℝ := y - 3
  x * y = -10 ∧ y' = -1 := by sorry

end translation_product_l2542_254298


namespace sum_of_last_two_digits_of_9_pow_207_l2542_254289

theorem sum_of_last_two_digits_of_9_pow_207 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 9^207 ≡ 10*a + b [ZMOD 100] ∧ a + b = 15 :=
sorry

end sum_of_last_two_digits_of_9_pow_207_l2542_254289


namespace stating_largest_cone_in_cube_l2542_254213

/-- Represents the dimensions of a cone carved from a cube. -/
structure ConeDimensions where
  height : ℝ
  baseRadius : ℝ
  volume : ℝ

/-- 
Theorem stating the dimensions of the largest cone that can be carved from a cube.
The cone's axis coincides with one of the cube's body diagonals.
-/
theorem largest_cone_in_cube (a : ℝ) (ha : a > 0) : 
  ∃ (cone : ConeDimensions), 
    cone.height = a * Real.sqrt 3 / 2 ∧
    cone.baseRadius = a * Real.sqrt 3 / (2 * Real.sqrt 2) ∧
    cone.volume = π * a^3 * Real.sqrt 3 / 16 ∧
    ∀ (other : ConeDimensions), other.volume ≤ cone.volume := by
  sorry

end stating_largest_cone_in_cube_l2542_254213


namespace solution_range_l2542_254204

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1 - m) * x = 2 - 3 * x) → m < 4 := by
sorry

end solution_range_l2542_254204


namespace probability_sum_five_l2542_254256

/-- The probability of obtaining a sum of 5 when rolling two dice of different sizes simultaneously -/
theorem probability_sum_five (total_outcomes : ℕ) (favorable_outcomes : ℕ) : 
  total_outcomes = 36 → favorable_outcomes = 4 → (favorable_outcomes : ℚ) / total_outcomes = 1 / 9 := by
  sorry

end probability_sum_five_l2542_254256


namespace p_or_q_is_true_l2542_254272

open Real

-- Define the statements p and q
def p : Prop := ∀ x, (deriv (λ x => 3 * x^2 + Real.log 3)) x = 6 * x + 3

def q : Prop := ∀ x, x ∈ Set.Ioo (-3 : ℝ) 1 ↔ 
  (deriv (λ x => (3 - x^2) * Real.exp x)) x > 0

-- Theorem statement
theorem p_or_q_is_true : p ∨ q := by sorry

end p_or_q_is_true_l2542_254272


namespace smallest_N_eight_works_smallest_N_is_8_l2542_254214

theorem smallest_N : ∀ N : ℕ+, 
  (∃ a b c d : ℕ, 
    a = N.val * 125 / 1000 ∧
    b = N.val * 500 / 1000 ∧
    c = N.val * 250 / 1000 ∧
    d = N.val * 125 / 1000) →
  N.val ≥ 8 :=
by sorry

theorem eight_works : 
  ∃ a b c d : ℕ,
    a = 8 * 125 / 1000 ∧
    b = 8 * 500 / 1000 ∧
    c = 8 * 250 / 1000 ∧
    d = 8 * 125 / 1000 :=
by sorry

theorem smallest_N_is_8 : 
  ∀ N : ℕ+, 
    (∃ a b c d : ℕ, 
      a = N.val * 125 / 1000 ∧
      b = N.val * 500 / 1000 ∧
      c = N.val * 250 / 1000 ∧
      d = N.val * 125 / 1000) ↔
    N.val ≥ 8 :=
by sorry

end smallest_N_eight_works_smallest_N_is_8_l2542_254214


namespace larger_number_problem_l2542_254245

theorem larger_number_problem (x y : ℝ) 
  (sum_eq : x + y = 147)
  (relation : x = 0.375 * y + 4)
  (x_larger : x > y) : 
  x = 43 := by
sorry

end larger_number_problem_l2542_254245


namespace tank_capacity_l2542_254230

theorem tank_capacity (bucket_capacity : ℚ) : 
  (13 * 42 = 91 * bucket_capacity) → bucket_capacity = 6 := by
  sorry

end tank_capacity_l2542_254230
