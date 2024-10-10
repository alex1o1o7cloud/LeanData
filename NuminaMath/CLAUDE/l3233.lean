import Mathlib

namespace quadratic_one_root_l3233_323325

theorem quadratic_one_root (m : ℝ) : 
  (∀ x : ℝ, x^2 - 8*m*x + 15*m = 0 → (∀ y : ℝ, y^2 - 8*m*y + 15*m = 0 → y = x)) → 
  m = 15/16 :=
sorry

end quadratic_one_root_l3233_323325


namespace sqrt_sum_equality_l3233_323352

theorem sqrt_sum_equality (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a*b + b*c + c*a = 0 ∧ a + b + c ≥ 0 :=
by sorry

end sqrt_sum_equality_l3233_323352


namespace correct_arrangement_count_l3233_323365

/-- The number of ways to arrange 7 people with specific adjacency conditions -/
def arrangement_count : ℕ := 960

/-- Proves that the number of arrangements is correct -/
theorem correct_arrangement_count : arrangement_count = 960 := by
  sorry

end correct_arrangement_count_l3233_323365


namespace large_triangle_perimeter_l3233_323350

/-- An isosceles triangle with two sides of length 20 and one side of length 10 -/
structure SmallTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : side2 = side3
  length_side1 : side1 = 10
  length_side2 : side2 = 20

/-- A triangle similar to SmallTriangle with shortest side of length 50 -/
structure LargeTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  shortest_side : side1 = 50
  similar_to_small : ∃ (k : ℝ), side1 = k * 10 ∧ side2 = k * 20 ∧ side3 = k * 20

/-- The perimeter of a triangle -/
def perimeter (t : LargeTriangle) : ℝ := t.side1 + t.side2 + t.side3

/-- Theorem stating that the perimeter of the larger triangle is 250 -/
theorem large_triangle_perimeter :
  ∀ (small : SmallTriangle) (large : LargeTriangle),
  perimeter large = 250 := by sorry

end large_triangle_perimeter_l3233_323350


namespace f_monotone_and_no_min_l3233_323377

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a - 1) * Real.exp (x - 1) - (1/2) * x^2 + a * x

theorem f_monotone_and_no_min (x : ℝ) (hx : x > 0) :
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f 1 x₁ < f 1 x₂) ∧
  (∃ a₁ a₂ : ℤ, (∀ x > 0, ∃ y > x, f a₁ y < f a₁ x) ∧
                (∀ x > 0, ∃ y > x, f a₂ y < f a₂ x) ∧
                a₁ + a₂ = 3) :=
by sorry

end f_monotone_and_no_min_l3233_323377


namespace workbook_problems_l3233_323357

theorem workbook_problems (P : ℚ) 
  (h1 : (1/2 : ℚ) * P + (1/4 : ℚ) * P + (1/6 : ℚ) * P + 20 = P) : 
  P = 240 :=
by sorry

end workbook_problems_l3233_323357


namespace katerina_weight_l3233_323349

theorem katerina_weight (total_weight : ℕ) (alexa_weight : ℕ) 
  (h1 : total_weight = 95)
  (h2 : alexa_weight = 46) :
  total_weight - alexa_weight = 49 := by
  sorry

end katerina_weight_l3233_323349


namespace sphere_in_cube_surface_area_l3233_323370

theorem sphere_in_cube_surface_area (cube_edge : ℝ) (h : cube_edge = 4) :
  let sphere_radius := cube_edge / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 16 * Real.pi :=
by sorry

end sphere_in_cube_surface_area_l3233_323370


namespace partial_fraction_decomposition_l3233_323335

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ), 
    (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
      (-2 * x^2 + 5 * x - 7) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) ∧
    P = 7 ∧ Q = -9 ∧ R = 5 := by
  sorry

end partial_fraction_decomposition_l3233_323335


namespace surjective_iff_coprime_l3233_323374

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- The function f(x) = x^x mod n -/
def f (n : ℕ) (x : ℕ+) : ZMod n :=
  (x : ZMod n) ^ (x : ℕ)

/-- Surjectivity of f -/
def is_surjective (n : ℕ) : Prop :=
  Function.Surjective (f n)

theorem surjective_iff_coprime (n : ℕ) (h : n > 0) :
  is_surjective n ↔ Nat.Coprime n (phi n) := by sorry

end surjective_iff_coprime_l3233_323374


namespace arithmetic_sequence_common_difference_l3233_323327

/-- An arithmetic sequence {a_n} with a_1 + a_9 = 10 and a_2 = -1 has a common difference of 2. -/
theorem arithmetic_sequence_common_difference : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 + a 9 = 10 →                     -- given condition
  a 2 = -1 →                           -- given condition
  a 2 - a 1 = 2 :=                     -- conclusion: common difference is 2
by
  sorry

end arithmetic_sequence_common_difference_l3233_323327


namespace sum_of_remainders_l3233_323359

theorem sum_of_remainders : Int.mod (Int.mod (5^(5^(5^5))) 500 + Int.mod (2^(2^(2^2))) 500) 500 = 49 := by
  sorry

end sum_of_remainders_l3233_323359


namespace only_statement_4_implies_negation_l3233_323310

theorem only_statement_4_implies_negation (p q : Prop) :
  -- Define the four statements
  let s1 := p ∨ q
  let s2 := p ∧ ¬q
  let s3 := ¬p ∨ q
  let s4 := ¬p ∧ ¬q
  -- Define the negation of "p or q is true"
  let neg_p_or_q := ¬(p ∨ q)
  -- The theorem: only s4 implies neg_p_or_q
  (s1 → neg_p_or_q) = False ∧
  (s2 → neg_p_or_q) = False ∧
  (s3 → neg_p_or_q) = False ∧
  (s4 → neg_p_or_q) = True :=
by
  sorry

#check only_statement_4_implies_negation

end only_statement_4_implies_negation_l3233_323310


namespace quadratic_inequality_range_l3233_323354

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a - 1)*x + (a - 1) > 0) ↔ (1 < a ∧ a < 5) := by
  sorry

end quadratic_inequality_range_l3233_323354


namespace perpendicular_condition_l3233_323390

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m : ℝ) : Prop :=
  (-m / (2*m - 1)) * (-3 / m) = -1

/-- The first line equation -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  m*x + (2*m - 1)*y + 1 = 0

/-- The second line equation -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  3*x + m*y + 2 = 0

/-- m = -1 is sufficient but not necessary for the lines to be perpendicular -/
theorem perpendicular_condition (m : ℝ) :
  (m = -1 → are_perpendicular m) ∧
  ¬(are_perpendicular m → m = -1) :=
sorry

end perpendicular_condition_l3233_323390


namespace square_difference_l3233_323337

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := by
  sorry

end square_difference_l3233_323337


namespace other_factor_proof_l3233_323344

theorem other_factor_proof (w : ℕ) (h1 : w > 0) 
  (h2 : ∃ k : ℕ, 936 * w = 2^5 * 13^2 * k) 
  (h3 : w ≥ 156) 
  (h4 : ∀ v : ℕ, v > 0 → v < 156 → ¬(∃ k : ℕ, 936 * v = 2^5 * 13^2 * k)) : 
  ∃ m : ℕ, w = 3 * m ∧ ∃ k : ℕ, 936 * m = 2^5 * 13^2 * k := by
sorry

end other_factor_proof_l3233_323344


namespace reading_time_proof_l3233_323364

/-- The number of days it took for Ryan and his brother to finish their books -/
def days_to_finish : ℕ := 7

/-- Ryan's total number of pages -/
def ryan_total_pages : ℕ := 2100

/-- Number of pages Ryan's brother reads per day -/
def brother_pages_per_day : ℕ := 200

/-- The difference in pages read per day between Ryan and his brother -/
def page_difference : ℕ := 100

theorem reading_time_proof :
  ryan_total_pages = (brother_pages_per_day + page_difference) * days_to_finish ∧
  ryan_total_pages % (brother_pages_per_day + page_difference) = 0 := by
  sorry

#check reading_time_proof

end reading_time_proof_l3233_323364


namespace temp_difference_l3233_323351

/-- The highest temperature in Xiangyang City on March 7, 2023 -/
def highest_temp : ℝ := 26

/-- The lowest temperature in Xiangyang City on March 7, 2023 -/
def lowest_temp : ℝ := 14

/-- The theorem states that the difference between the highest and lowest temperatures is 12°C -/
theorem temp_difference : highest_temp - lowest_temp = 12 := by
  sorry

end temp_difference_l3233_323351


namespace election_votes_theorem_l3233_323304

/-- Represents an election between two candidates -/
structure Election where
  total_votes : ℕ
  winner_votes : ℕ
  loser_votes : ℕ

/-- The conditions of the election problem -/
def election_conditions (e : Election) : Prop :=
  e.winner_votes + e.loser_votes = e.total_votes ∧
  e.winner_votes - e.loser_votes = (e.total_votes : ℚ) * (1 / 10) ∧
  (e.winner_votes - 1500) - (e.loser_votes + 1500) = -(e.total_votes : ℚ) * (1 / 10)

/-- The theorem stating that under the given conditions, the total votes is 15000 -/
theorem election_votes_theorem (e : Election) :
  election_conditions e → e.total_votes = 15000 := by
  sorry

end election_votes_theorem_l3233_323304


namespace hannah_stocking_stuffers_l3233_323398

/-- The number of candy canes per stocking -/
def candy_canes_per_stocking : ℕ := 4

/-- The number of beanie babies per stocking -/
def beanie_babies_per_stocking : ℕ := 2

/-- The number of books per stocking -/
def books_per_stocking : ℕ := 1

/-- The number of kids Hannah has -/
def number_of_kids : ℕ := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stocking_stuffers : ℕ :=
  (candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking) * number_of_kids

theorem hannah_stocking_stuffers :
  total_stocking_stuffers = 21 := by
  sorry

end hannah_stocking_stuffers_l3233_323398


namespace smallest_gcd_of_b_c_l3233_323316

theorem smallest_gcd_of_b_c (a b c x y : ℕ+) 
  (hab : Nat.gcd a b = 120)
  (hac : Nat.gcd a c = 1001)
  (hb : b = 120 * x)
  (hc : c = 1001 * y) :
  ∃ (b' c' : ℕ+), Nat.gcd b' c' = 1 ∧ 
    ∀ (b'' c'' : ℕ+), (∃ (x'' y'' : ℕ+), b'' = 120 * x'' ∧ c'' = 1001 * y'') → 
      Nat.gcd b'' c'' ≥ Nat.gcd b' c' :=
sorry

end smallest_gcd_of_b_c_l3233_323316


namespace arithmetic_sequence_solution_l3233_323317

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (d : ℤ) :
  is_arithmetic_sequence a d →
  a 3 * a 7 = -16 →
  a 4 + a 6 = 0 →
  ((a 1 = -8 ∧ d = 2) ∨ (a 1 = 8 ∧ d = -2)) :=
by sorry

end arithmetic_sequence_solution_l3233_323317


namespace fred_weekend_earnings_l3233_323321

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings : ℕ := 16

/-- Fred's earnings from washing cars -/
def car_wash_earnings : ℕ := 74

/-- Fred's total earnings over the weekend -/
def total_earnings : ℕ := newspaper_earnings + car_wash_earnings

/-- Theorem stating that Fred's total earnings over the weekend equal $90 -/
theorem fred_weekend_earnings : total_earnings = 90 := by
  sorry

end fred_weekend_earnings_l3233_323321


namespace g_of_one_eq_neg_25_l3233_323368

/-- g is a rational function satisfying the given equation for all non-zero x -/
def g_equation (g : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * g (2 / x) + 3 * g x / x = 2 * x^3 - x

/-- Theorem: If g satisfies the equation, then g(1) = -25 -/
theorem g_of_one_eq_neg_25 (g : ℚ → ℚ) (h : g_equation g) : g 1 = -25 := by
  sorry

end g_of_one_eq_neg_25_l3233_323368


namespace quadratic_root_range_l3233_323395

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x, a * x^2 + (a + 1) * x + 6 * a = 0) ∧ 
  (x₁ ≠ x₂) ∧ 
  (x₁ < 1 ∧ 1 < x₂) ∧
  (a * x₁^2 + (a + 1) * x₁ + 6 * a = 0) ∧
  (a * x₂^2 + (a + 1) * x₂ + 6 * a = 0) →
  -1/8 < a ∧ a < 0 :=
by sorry

end quadratic_root_range_l3233_323395


namespace sum_of_reciprocal_relations_l3233_323332

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1 / x + 1 / y = 1) 
  (h4 : 1 / x - 1 / y = 5) : 
  x + y = -1/6 := by
  sorry

end sum_of_reciprocal_relations_l3233_323332


namespace det_A_eq_31_l3233_323386

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 5; 3, 6, -2; 1, -1, 3]

theorem det_A_eq_31 : Matrix.det A = 31 := by
  sorry

end det_A_eq_31_l3233_323386


namespace not_all_squares_congruent_square_is_convex_square_is_equiangular_square_has_equal_sides_all_squares_are_similar_l3233_323393

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

-- Other properties of squares (not used in the proof, but included for completeness)
theorem square_is_convex (s : Square) : True := by
  sorry

theorem square_is_equiangular (s : Square) : True := by
  sorry

theorem square_has_equal_sides (s : Square) : True := by
  sorry

theorem all_squares_are_similar : ∀ (s1 s2 : Square), True := by
  sorry

end not_all_squares_congruent_square_is_convex_square_is_equiangular_square_has_equal_sides_all_squares_are_similar_l3233_323393


namespace problem_solution_l3233_323387

theorem problem_solution : ∃ x : ℝ, (5 * 12) / (180 / 3) + x = 65 ∧ x = 64 := by
  sorry

end problem_solution_l3233_323387


namespace max_intersecting_chords_2017_l3233_323381

/-- Given a circle with n distinct points, this function calculates the maximum number of chords
    intersecting a line through one point, not passing through any other points. -/
def max_intersecting_chords (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  k * (n - 1 - k) + (n - 1)

/-- The theorem states that for a circle with 2017 points, the maximum number of
    intersecting chords is 1018080. -/
theorem max_intersecting_chords_2017 :
  max_intersecting_chords 2017 = 1018080 := by sorry

end max_intersecting_chords_2017_l3233_323381


namespace paul_pencil_days_l3233_323343

/-- Calculates the number of days Paul makes pencils in a week -/
def pencil_making_days (
  pencils_per_day : ℕ) 
  (initial_stock : ℕ) 
  (pencils_sold : ℕ) 
  (final_stock : ℕ) : ℕ :=
  (final_stock + pencils_sold - initial_stock) / pencils_per_day

theorem paul_pencil_days : 
  pencil_making_days 100 80 350 230 = 5 := by sorry

end paul_pencil_days_l3233_323343


namespace trigonometric_equation_l3233_323314

theorem trigonometric_equation (x : ℝ) :
  (1 / Real.cos (2022 * x) + Real.tan (2022 * x) = 1 / 2022) →
  (1 / Real.cos (2022 * x) - Real.tan (2022 * x) = 2022) := by
  sorry

end trigonometric_equation_l3233_323314


namespace f_composition_half_l3233_323397

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_half : f (f (1/2)) = 1/2 := by
  sorry

end f_composition_half_l3233_323397


namespace cone_base_diameter_l3233_323309

theorem cone_base_diameter (r : ℝ) (h1 : r > 0) : 
  (π * r^2 + π * r * (2 * r) = 3 * π) → 
  (2 * r = 2) := by
sorry

end cone_base_diameter_l3233_323309


namespace rick_has_two_sisters_l3233_323339

/-- Calculates the number of Rick's sisters based on the given card distribution. -/
def number_of_sisters (total_cards : ℕ) (kept_cards : ℕ) (miguel_cards : ℕ) 
  (friends : ℕ) (cards_per_friend : ℕ) (cards_per_sister : ℕ) : ℕ :=
  let remaining_cards := total_cards - kept_cards - miguel_cards - (friends * cards_per_friend)
  remaining_cards / cards_per_sister

/-- Theorem stating that Rick has 2 sisters given the card distribution. -/
theorem rick_has_two_sisters :
  number_of_sisters 130 15 13 8 12 3 = 2 := by
  sorry

end rick_has_two_sisters_l3233_323339


namespace intersection_distance_l3233_323312

-- Define the line x = 4
def line (x : ℝ) : Prop := x = 4

-- Define the curve x = t², y = t³
def curve (t x y : ℝ) : Prop := x = t^2 ∧ y = t^3

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 ∧ curve 2 A.1 A.2 ∧
  line B.1 ∧ curve (-2) B.1 B.2

-- Theorem statement
theorem intersection_distance :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 :=
by sorry

end intersection_distance_l3233_323312


namespace dice_rotation_probability_l3233_323356

/-- The number of faces on a die -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 3

/-- The total number of ways to paint a single die -/
def ways_to_paint_one_die : ℕ := num_colors ^ num_faces

/-- The total number of ways to paint two dice -/
def total_paint_combinations : ℕ := ways_to_paint_one_die ^ 2

/-- The number of ways two dice can appear identical after rotation -/
def identical_after_rotation : ℕ := 1119

/-- The probability that two independently painted dice appear identical after rotation -/
theorem dice_rotation_probability :
  (identical_after_rotation : ℚ) / total_paint_combinations = 1119 / 531441 := by
  sorry

end dice_rotation_probability_l3233_323356


namespace min_length_intersection_l3233_323328

/-- The minimum length of the intersection of two sets M and N -/
theorem min_length_intersection (m n : ℝ) : 
  0 ≤ m → m + 3/4 ≤ 1 → 0 ≤ n - 1/3 → n ≤ 1 → 
  ∃ (a b : ℝ), 
    (∀ x, x ∈ (Set.Icc m (m + 3/4) ∩ Set.Icc (n - 1/3) n) → a ≤ x ∧ x ≤ b) ∧
    b - a = 1/12 ∧
    (∀ c d, (∀ x, x ∈ (Set.Icc m (m + 3/4) ∩ Set.Icc (n - 1/3) n) → c ≤ x ∧ x ≤ d) → 
      d - c ≥ 1/12) :=
sorry

end min_length_intersection_l3233_323328


namespace select_shoes_result_l3233_323353

/-- The number of ways to select 4 individual shoes from 5 pairs of shoes,
    such that exactly 1 pair is among the selected shoes -/
def select_shoes (total_pairs : ℕ) (shoes_to_select : ℕ) : ℕ :=
  total_pairs * (total_pairs - 1).choose 2 * 2 * 2

/-- Theorem stating that the number of ways to select 4 individual shoes
    from 5 pairs of shoes, such that exactly 1 pair is among them, is 120 -/
theorem select_shoes_result :
  select_shoes 5 4 = 120 := by sorry

end select_shoes_result_l3233_323353


namespace largest_m_for_factorization_l3233_323305

def is_valid_factorization (m A B : ℤ) : Prop :=
  A * B = 90 ∧ 5 * B + A = m

theorem largest_m_for_factorization :
  (∃ (m : ℤ), ∀ (A B : ℤ), is_valid_factorization m A B →
    ∀ (m' : ℤ), (∃ (A' B' : ℤ), is_valid_factorization m' A' B') → m' ≤ m) ∧
  (∃ (A B : ℤ), is_valid_factorization 451 A B) :=
sorry

end largest_m_for_factorization_l3233_323305


namespace isosceles_triangle_side_length_l3233_323360

/-- An isosceles triangle with an angle bisector dividing the perimeter -/
structure IsoscelesTriangleWithBisector where
  /-- The length of one of the equal sides -/
  side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- The length of the angle bisector -/
  bisector : ℝ
  /-- The angle bisector divides the perimeter into parts of 63 and 35 -/
  perimeter_division : side + bisector = 63 ∧ side + base / 2 = 35
  /-- The triangle is isosceles -/
  isosceles : side > 0

/-- The length of the equal sides in the given isosceles triangle is not 26.4, 33, or 38.5 -/
theorem isosceles_triangle_side_length
  (t : IsoscelesTriangleWithBisector) :
  t.side ≠ 26.4 ∧ t.side ≠ 33 ∧ t.side ≠ 38.5 := by
  sorry

end isosceles_triangle_side_length_l3233_323360


namespace otimes_property_implies_a_range_l3233_323330

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem otimes_property_implies_a_range :
  (∀ x : ℝ, otimes x (x + a) < 1) → -1 < a ∧ a < 3 :=
sorry

end otimes_property_implies_a_range_l3233_323330


namespace sqrt_one_fourth_l3233_323367

theorem sqrt_one_fourth : Real.sqrt (1 / 4) = 1 / 2 := by
  sorry

end sqrt_one_fourth_l3233_323367


namespace cookies_left_l3233_323384

theorem cookies_left (whole_cookies : ℕ) (greg_ate : ℕ) (brad_ate : ℕ) : 
  whole_cookies = 14 → greg_ate = 4 → brad_ate = 6 → 
  whole_cookies * 2 - (greg_ate + brad_ate) = 18 := by
  sorry

end cookies_left_l3233_323384


namespace inscribed_circle_radius_l3233_323376

/-- A quadrilateral circumscribed around a circle -/
structure CircumscribedQuadrilateral where
  /-- The sum of two opposite sides -/
  opposite_sides_sum : ℝ
  /-- The area of the quadrilateral -/
  area : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ

/-- Theorem: If the sum of opposite sides is 10 and the area is 12, 
    then the radius of the inscribed circle is 6/5 -/
theorem inscribed_circle_radius 
  (q : CircumscribedQuadrilateral) 
  (h1 : q.opposite_sides_sum = 10) 
  (h2 : q.area = 12) : 
  q.inradius = 6/5 := by
  sorry

end inscribed_circle_radius_l3233_323376


namespace triangle_inequality_holds_l3233_323396

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def triangle_sides (x : ℕ) : ℕ × ℕ × ℕ :=
  (6, x + 3, 2 * x - 1)

theorem triangle_inequality_holds (x : ℕ) :
  (∃ (y : ℕ), y ∈ ({2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧ x = y) ↔
  (let (a, b, c) := triangle_sides x
   is_valid_triangle a b c ∧ x > 0) :=
sorry

end triangle_inequality_holds_l3233_323396


namespace cds_on_shelf_l3233_323303

/-- The number of CDs that a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- The number of racks that can fit on a shelf -/
def racks_per_shelf : ℕ := 4

/-- Theorem stating the total number of CDs that can fit on a shelf -/
theorem cds_on_shelf : cds_per_rack * racks_per_shelf = 32 := by
  sorry

end cds_on_shelf_l3233_323303


namespace one_fourth_greater_than_one_fifth_of_successor_l3233_323346

theorem one_fourth_greater_than_one_fifth_of_successor :
  let N : ℝ := 24.000000000000004
  (1/4 : ℝ) * N - (1/5 : ℝ) * (N + 1) = 1.000000000000000 := by
  sorry

end one_fourth_greater_than_one_fifth_of_successor_l3233_323346


namespace distance_to_asymptote_l3233_323334

def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

def point : ℝ × ℝ := (3, 0)

theorem distance_to_asymptote :
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), hyperbola x y → (a * x + b * y + c = 0 ∨ a * x + b * y - c = 0)) ∧
    (|a * point.1 + b * point.2 + c| / Real.sqrt (a^2 + b^2) = 9/5) :=
sorry

end distance_to_asymptote_l3233_323334


namespace andrews_eggs_l3233_323355

-- Define the costs
def toast_cost : ℕ := 1
def egg_cost : ℕ := 3

-- Define Dale's breakfast
def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

-- Define Andrew's breakfast
def andrew_toast : ℕ := 1

-- Define the total cost
def total_cost : ℕ := 15

-- Theorem to prove
theorem andrews_eggs :
  ∃ (andrew_eggs : ℕ),
    toast_cost * (dale_toast + andrew_toast) +
    egg_cost * (dale_eggs + andrew_eggs) = total_cost ∧
    andrew_eggs = 2 := by
  sorry

end andrews_eggs_l3233_323355


namespace f_inequality_l3233_323301

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem f_inequality (a : ℝ) (h : a ≤ -2) :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → |f a x₁ - f a x₂| ≥ 4 * |x₁ - x₂| := by
  sorry

end f_inequality_l3233_323301


namespace train_distance_difference_l3233_323302

theorem train_distance_difference (v1 v2 d : ℝ) (hv1 : v1 = 20) (hv2 : v2 = 25) (hd : d = 495) :
  let t := d / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d2 - d1 = 55 := by sorry

end train_distance_difference_l3233_323302


namespace boat_speed_ratio_l3233_323306

/-- Proves that the ratio of downstream to upstream speed is 2:1 for a boat in a river --/
theorem boat_speed_ratio (v : ℝ) : 
  v > 3 →  -- Boat speed must be greater than river flow
  (4 / (v + 3) + 4 / (v - 3) = 1) →  -- Total travel time is 1 hour
  ((v + 3) / (v - 3) = 2) :=  -- Ratio of downstream to upstream speed
by
  sorry

#check boat_speed_ratio

end boat_speed_ratio_l3233_323306


namespace line_plane_relationship_l3233_323326

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (m n : Line) (α β : Plane) 
  (h1 : parallel α β) 
  (h2 : perpendicular m α) 
  (h3 : perpendicular_lines m n) :
  contained_in n β ∨ parallel_line_plane n β :=
sorry

end line_plane_relationship_l3233_323326


namespace inequality_system_solution_range_l3233_323345

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (2 * ↑x - 1 > 3 ∧ ↑x ≤ 2 * a - 1) ∧
    (2 * ↑y - 1 > 3 ∧ ↑y ≤ 2 * a - 1) ∧
    (2 * ↑z - 1 > 3 ∧ ↑z ≤ 2 * a - 1) ∧
    (∀ (w : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z → ¬(2 * ↑w - 1 > 3 ∧ ↑w ≤ 2 * a - 1))) →
  (3 ≤ a ∧ a < 3.5) :=
by sorry

end inequality_system_solution_range_l3233_323345


namespace a_range_for_g_three_zeros_l3233_323308

open Real

noncomputable def f (a b x : ℝ) : ℝ := exp x - 2 * (a - 1) * x - b

noncomputable def g (a b x : ℝ) : ℝ := exp x - (a - 1) * x^2 - b * x - 1

theorem a_range_for_g_three_zeros (a b : ℝ) :
  (g a b 1 = 0) →
  (∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < 1 ∧
    g a b x₁ = 0 ∧ g a b x₂ = 0 ∧ g a b x₃ = 0) →
  (e - 1 < a ∧ a < 2) :=
by sorry

end a_range_for_g_three_zeros_l3233_323308


namespace abs_z_equals_5_sqrt_2_l3233_323323

theorem abs_z_equals_5_sqrt_2 (z : ℂ) (h : z^2 = -48 + 14*I) : Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end abs_z_equals_5_sqrt_2_l3233_323323


namespace arithmetic_sequence_max_product_l3233_323369

/-- An arithmetic sequence with 1990 terms -/
def ArithmeticSequence := Fin 1990 → ℝ

/-- The common difference of an arithmetic sequence -/
def commonDifference (a : ArithmeticSequence) : ℝ :=
  a 1 - a 0

/-- The condition that all terms in the sequence are positive -/
def allPositive (a : ArithmeticSequence) : Prop :=
  ∀ i j : Fin 1990, a i * a j > 0

/-- The b_k sequence defined in the problem -/
def b (a : ArithmeticSequence) (k : Fin 1990) : ℝ :=
  a k * a (1989 - k)

theorem arithmetic_sequence_max_product 
  (a : ArithmeticSequence) 
  (hd : commonDifference a ≠ 0) 
  (hp : allPositive a) : 
  (∀ k : Fin 1990, b a k ≤ b a 994 ∨ b a k ≤ b a 995) :=
sorry

end arithmetic_sequence_max_product_l3233_323369


namespace larger_integer_problem_l3233_323385

theorem larger_integer_problem (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 7 / 3 → 
  a * b = 189 → 
  max a b = 21 := by
sorry

end larger_integer_problem_l3233_323385


namespace refrigeratorSample_is_valid_l3233_323388

/-- Represents a systematic sample -/
structure SystematicSample (N : ℕ) (n : ℕ) where
  start : ℕ
  sequence : Fin n → ℕ
  valid : ∀ i : Fin n, sequence i = start + i.val * (N / n)

/-- The specific systematic sample for the refrigerator problem -/
def refrigeratorSample : SystematicSample 60 6 :=
  { start := 3,
    sequence := λ i => 3 + i.val * 10,
    valid := sorry }

/-- Theorem stating that the refrigeratorSample is valid -/
theorem refrigeratorSample_is_valid :
  ∀ i : Fin 6, refrigeratorSample.sequence i ≤ 60 :=
by sorry

end refrigeratorSample_is_valid_l3233_323388


namespace elrond_arwen_tulip_ratio_l3233_323391

/-- Given that Arwen picked 20 tulips and the total number of tulips picked by Arwen and Elrond is 60,
    prove that the ratio of Elrond's tulips to Arwen's tulips is 2:1 -/
theorem elrond_arwen_tulip_ratio :
  let arwen_tulips : ℕ := 20
  let total_tulips : ℕ := 60
  let elrond_tulips : ℕ := total_tulips - arwen_tulips
  (elrond_tulips : ℚ) / (arwen_tulips : ℚ) = 2 / 1 := by
  sorry

end elrond_arwen_tulip_ratio_l3233_323391


namespace double_sum_of_factors_17_l3233_323361

/-- The sum of positive factors of a natural number -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- The boxed notation representing the sum of positive factors -/
notation "⌈" n "⌉" => sum_of_factors n

/-- Theorem stating that the double application of sum_of_factors to 17 equals 39 -/
theorem double_sum_of_factors_17 : ⌈⌈17⌉⌉ = 39 := by sorry

end double_sum_of_factors_17_l3233_323361


namespace science_class_students_l3233_323362

theorem science_class_students :
  ∃! n : ℕ, 0 < n ∧ n < 60 ∧ n % 6 = 4 ∧ n % 8 = 5 ∧ n = 46 := by
sorry

end science_class_students_l3233_323362


namespace residue_theorem_l3233_323318

theorem residue_theorem (m k : ℕ) (hm : m > 0) (hk : k > 0) :
  (Nat.gcd m k = 1 →
    ∃ (a b : ℕ → ℕ),
      ∀ i j s t, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ k ∧
                 1 ≤ s ∧ s ≤ m ∧ 1 ≤ t ∧ t ≤ k ∧
                 (i ≠ s ∨ j ≠ t) →
                 (a i * b j) % (m * k) ≠ (a s * b t) % (m * k)) ∧
  (Nat.gcd m k > 1 →
    ∀ (a b : ℕ → ℕ),
      ∃ i j s t, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ k ∧
                 1 ≤ s ∧ s ≤ m ∧ 1 ≤ t ∧ t ≤ k ∧
                 (i ≠ s ∨ j ≠ t) ∧
                 (a i * b j) % (m * k) = (a s * b t) % (m * k)) :=
by sorry

end residue_theorem_l3233_323318


namespace negation_of_universal_proposition_l3233_323320

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l3233_323320


namespace samara_oil_spending_l3233_323322

/-- The amount Alberto spent on his car -/
def alberto_spent : ℕ := 2457

/-- The amount Samara spent on tires -/
def samara_tires : ℕ := 467

/-- The amount Samara spent on detailing -/
def samara_detailing : ℕ := 79

/-- The difference between Alberto's and Samara's spending -/
def spending_difference : ℕ := 1886

/-- The amount Samara spent on oil -/
def samara_oil : ℕ := 25

theorem samara_oil_spending : 
  alberto_spent = samara_oil + samara_tires + samara_detailing + spending_difference :=
by sorry

end samara_oil_spending_l3233_323322


namespace small_cubes_to_large_cube_l3233_323333

theorem small_cubes_to_large_cube (large_volume small_volume : ℕ) 
  (h : large_volume = 1000 ∧ small_volume = 8) : 
  (large_volume / small_volume : ℕ) = 125 := by
  sorry

end small_cubes_to_large_cube_l3233_323333


namespace equal_area_parallelograms_locus_l3233_323373

/-- Given a triangle ABC and an interior point P, this theorem states that if the areas of
    parallelograms GPDC and FPEB (formed by lines parallel to the sides through P) are equal,
    then P lies on a specific line. -/
theorem equal_area_parallelograms_locus (a b c k l : ℝ) :
  let A : ℝ × ℝ := (0, a)
  let B : ℝ × ℝ := (-b, 0)
  let C : ℝ × ℝ := (c, 0)
  let P : ℝ × ℝ := (k, l)
  let E : ℝ × ℝ := (k - b*l/a, 0)
  let D : ℝ × ℝ := (k + l*c/a, 0)
  let F : ℝ × ℝ := (b*l/a - b, l)
  let G : ℝ × ℝ := (c - l*c/a, l)
  a > 0 ∧ b > 0 ∧ c > 0 ∧ k > -b ∧ k < c ∧ l > 0 ∧ l < a →
  abs (l/2 * (-c + 2*l*c/a)) = abs (l/2 * (-b + 2*l*b/a)) →
  2*a*k + (c - b)*l + a*(b - c) = 0 :=
sorry

end equal_area_parallelograms_locus_l3233_323373


namespace local_minimum_implies_c_equals_two_l3233_323382

/-- The function f(x) defined as x(x - c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) --/
def f_deriv (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem local_minimum_implies_c_equals_two :
  ∀ c : ℝ, (∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f c x ≥ f c 2) →
  f_deriv c 2 = 0 →
  c = 2 :=
sorry

end local_minimum_implies_c_equals_two_l3233_323382


namespace xyz_product_l3233_323378

theorem xyz_product (x y z : ℚ) 
  (eq1 : x + y + z = 1)
  (eq2 : x + y - z = 2)
  (eq3 : x - y - z = 3) :
  x * y * z = 1/2 := by
  sorry

end xyz_product_l3233_323378


namespace right_triangle_bisector_inscribed_circle_l3233_323319

/-- 
Theorem: In a right triangle with an inscribed circle of radius ρ 
and an angle bisector of length f for one of its acute angles, 
the condition f > √(8ρ) must hold.
-/
theorem right_triangle_bisector_inscribed_circle 
  (ρ f : ℝ) 
  (h_positive_ρ : ρ > 0) 
  (h_positive_f : f > 0) 
  (h_right_triangle : ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    (a * b) / (a + b + c) = ρ ∧
    f = (2 * a * b) / (a + b)) :
  f > Real.sqrt (8 * ρ) := by
sorry

end right_triangle_bisector_inscribed_circle_l3233_323319


namespace repeating_decimal_equals_two_thirds_l3233_323338

/-- The infinite repeating decimal 0.666... -/
def repeating_decimal : ℚ := 0.6666666666666667

/-- The theorem stating that the repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_decimal_equals_two_thirds : repeating_decimal = 2/3 := by
  sorry

end repeating_decimal_equals_two_thirds_l3233_323338


namespace negation_of_proposition_l3233_323399

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x y : ℝ, x^2 + y^2 - 1 > 0)) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) := by
  sorry

end negation_of_proposition_l3233_323399


namespace second_number_calculation_l3233_323313

theorem second_number_calculation (A B : ℝ) : 
  A = 700 → 
  0.3 * A = 0.6 * B + 120 → 
  B = 150 :=
by
  sorry

end second_number_calculation_l3233_323313


namespace b_minus_a_value_l3233_323315

theorem b_minus_a_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 2) (h3 : a + b > 0) :
  b - a = -1 ∨ b - a = -5 := by
  sorry

end b_minus_a_value_l3233_323315


namespace polygon_sides_l3233_323372

theorem polygon_sides (sum_interior_angles : ℝ) (n : ℕ) : 
  sum_interior_angles = 1260 → (n - 2) * 180 = sum_interior_angles → n = 9 := by
  sorry

end polygon_sides_l3233_323372


namespace max_cylinder_volume_l3233_323307

/-- The maximum volume of a cylinder formed by rotating a rectangle with perimeter 20cm around one of its edges -/
theorem max_cylinder_volume : 
  ∃ (V : ℝ), V = (4000 / 27) * Real.pi ∧ 
  (∀ (x : ℝ), 0 < x → x < 10 → 
    π * x^2 * (10 - x) ≤ V) :=
by sorry

end max_cylinder_volume_l3233_323307


namespace parallel_lines_coefficient_l3233_323300

theorem parallel_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, x + 2*a*y - 1 = 0 ↔ (3*a - 1)*x - a*y - 1 = 0) → 
  (a = 0 ∨ a = 1/6) :=
by sorry

end parallel_lines_coefficient_l3233_323300


namespace no_solution_implies_a_range_l3233_323347

theorem no_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| + |x - 2| < a)) → a ∈ Set.Ici 3 := by
  sorry

end no_solution_implies_a_range_l3233_323347


namespace function_satisfies_equation_l3233_323383

/-- Prove that for y = e^(x + x^2) + 2e^x, the equation y' - y = 2x e^(x + x^2) holds. -/
theorem function_satisfies_equation (x : ℝ) : 
  let y := Real.exp (x + x^2) + 2 * Real.exp x
  let y' := Real.exp (x + x^2) * (1 + 2*x) + 2 * Real.exp x
  y' - y = 2 * x * Real.exp (x + x^2) := by
sorry


end function_satisfies_equation_l3233_323383


namespace cinema_sampling_method_l3233_323363

/-- Represents a seating arrangement in a cinema --/
structure CinemaSeating where
  rows : ℕ
  seats_per_row : ℕ
  all_seats_filled : Bool

/-- Represents a sampling method --/
inductive SamplingMethod
  | LotteryMethod
  | RandomNumberTable
  | SystematicSampling
  | SamplingWithReplacement

/-- Defines the characteristics of systematic sampling --/
def is_systematic_sampling (seating : CinemaSeating) (selected_seat : ℕ) : Prop :=
  seating.all_seats_filled ∧
  selected_seat > 0 ∧
  selected_seat ≤ seating.seats_per_row ∧
  seating.rows > 1

/-- The main theorem to prove --/
theorem cinema_sampling_method (seating : CinemaSeating) (selected_seat : ℕ) :
  seating.rows = 50 →
  seating.seats_per_row = 60 →
  seating.all_seats_filled = true →
  selected_seat = 18 →
  is_systematic_sampling seating selected_seat →
  SamplingMethod.SystematicSampling = SamplingMethod.SystematicSampling :=
by
  sorry

end cinema_sampling_method_l3233_323363


namespace bounded_sequence_from_constrained_function_l3233_323389

def is_bounded_sequence (a : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ n : ℕ, |a n| ≤ M

theorem bounded_sequence_from_constrained_function
  (f : ℝ → ℝ)
  (hf_diff : Differentiable ℝ f)
  (hf_cont : Continuous (deriv f))
  (hf_bound : ∀ x : ℝ, 0 ≤ |deriv f x| ∧ |deriv f x| ≤ (1 : ℝ) / 2)
  (a : ℕ → ℝ)
  (ha_init : a 1 = 1)
  (ha_rec : ∀ n : ℕ, a (n + 1) = f (a n)) :
  is_bounded_sequence a :=
by
  sorry

end bounded_sequence_from_constrained_function_l3233_323389


namespace sequence_product_l3233_323394

theorem sequence_product : 
  (1/4) * 16 * (1/64) * 256 * (1/1024) * 4096 * (1/16384) * 65536 = 256 := by
  sorry

end sequence_product_l3233_323394


namespace donation_ratio_l3233_323380

def charity_raffle_problem (total_prize donation hotdog_cost leftover : ℕ) : Prop :=
  total_prize = donation + hotdog_cost + leftover ∧
  total_prize = 114 ∧
  hotdog_cost = 2 ∧
  leftover = 55

theorem donation_ratio (total_prize donation hotdog_cost leftover : ℕ) :
  charity_raffle_problem total_prize donation hotdog_cost leftover →
  (donation : ℚ) / total_prize = 55 / 114 := by
sorry

end donation_ratio_l3233_323380


namespace pythagorean_triple_sequence_l3233_323311

theorem pythagorean_triple_sequence (k : ℕ+) :
  ∃ (c : ℕ), (k * (2 * k - 2))^2 + (2 * k - 1)^2 = c^2 := by
  sorry

end pythagorean_triple_sequence_l3233_323311


namespace min_value_x_plus_3y_l3233_323336

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y + x*y = 9) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3*b + a*b = 9 → x + 3*y ≤ a + 3*b :=
by sorry

end min_value_x_plus_3y_l3233_323336


namespace fraction_equality_l3233_323348

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : s / u = 8 / 15) : 
  (2 * p * s - 3 * q * u) / (5 * q * u - 4 * p * s) = -5 / 7 := by
  sorry

end fraction_equality_l3233_323348


namespace intersection_of_M_and_N_l3233_323371

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - 3 * x)}
def N : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1/3} := by sorry

end intersection_of_M_and_N_l3233_323371


namespace division_problem_l3233_323342

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 109)
  (h2 : divisor = 12)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 := by
sorry

end division_problem_l3233_323342


namespace sphere_surface_area_from_volume_l3233_323379

theorem sphere_surface_area_from_volume :
  ∀ (r : ℝ),
  (4 / 3 : ℝ) * π * r^3 = 72 * π →
  4 * π * r^2 = 36 * π * 2^(2/3) :=
by
  sorry

end sphere_surface_area_from_volume_l3233_323379


namespace number_problem_l3233_323340

theorem number_problem (x : ℚ) : (x = (3/8) * x + 40) → x = 64 := by
  sorry

end number_problem_l3233_323340


namespace p_necessary_not_sufficient_for_q_l3233_323331

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x^2 - x - 2 < 0 → |x| < 2) ∧
  (∃ x : ℝ, |x| < 2 ∧ x^2 - x - 2 ≥ 0) := by
  sorry

end p_necessary_not_sufficient_for_q_l3233_323331


namespace factorial_sum_units_digit_l3233_323366

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_units_digit :
  ∀ n ≥ 99, units_digit (factorial_sum n) = 7 :=
by sorry

end factorial_sum_units_digit_l3233_323366


namespace sum_of_three_numbers_l3233_323392

theorem sum_of_three_numbers : 2.75 + 0.003 + 0.158 = 2.911 := by
  sorry

end sum_of_three_numbers_l3233_323392


namespace z_in_first_quadrant_l3233_323341

theorem z_in_first_quadrant (z : ℂ) (h : z * (1 - 2*I) = 3 - I) : 
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end z_in_first_quadrant_l3233_323341


namespace complement_A_B_l3233_323329

-- Define the set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (-2) 4, y = |x + 1|}

-- Define the set B
def B : Set ℝ := Set.Ici 2 ∩ Set.Iio 5

-- Theorem statement
theorem complement_A_B : 
  (Set.compl B) ∩ A = Set.Icc 0 2 ∪ {5} :=
sorry

end complement_A_B_l3233_323329


namespace tangent_angle_inclination_l3233_323375

/-- The angle of inclination of the tangent to y = (1/3)x³ - 2 at (1, -5/3) is 45° --/
theorem tangent_angle_inclination (f : ℝ → ℝ) (x : ℝ) :
  f x = (1/3) * x^3 - 2 →
  (deriv f) x = x^2 →
  x = 1 →
  f x = -5/3 →
  Real.arctan ((deriv f) x) = π/4 :=
by sorry

end tangent_angle_inclination_l3233_323375


namespace range_of_a_l3233_323324

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- Define the property of f not being monotonic
def not_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, x < y ∧ y < z ∧ (f x < f y ∧ f y > f z ∨ f x > f y ∧ f y < f z)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  not_monotonic (f a) ↔ a < -Real.sqrt 3 ∨ a > Real.sqrt 3 :=
sorry

end range_of_a_l3233_323324


namespace expand_and_simplify_l3233_323358

theorem expand_and_simplify (x : ℝ) : 3 * (x - 3) * (x + 10) + 2 * x = 3 * x^2 + 23 * x - 90 := by
  sorry

end expand_and_simplify_l3233_323358
