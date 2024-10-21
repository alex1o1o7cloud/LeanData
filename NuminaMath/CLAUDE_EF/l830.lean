import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_ratio_for_given_profit_l830_83017

/-- Given a profit percent, calculate the ratio of cost price to selling price -/
noncomputable def cost_price_to_selling_price_ratio (profit_percent : ℝ) : ℝ :=
  1 / (1 + profit_percent / 100)

/-- Theorem stating that for a profit percent of 21.951219512195124%, 
    the ratio of cost price to selling price is approximately 0.82 -/
theorem cost_price_ratio_for_given_profit :
  let profit_percent : ℝ := 21.951219512195124
  abs (cost_price_to_selling_price_ratio profit_percent - 0.82) < 0.001 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_ratio_for_given_profit_l830_83017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_drawers_l830_83032

def number_of_distributions (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

theorem balls_in_drawers (n k : ℕ) (h1 : n = 8) (h2 : k = 5) :
  number_of_distributions n k = 35 := by
  rw [number_of_distributions]
  rw [h1, h2]
  -- The calculation is: C(7,4) = 7! / (4! * 3!) = 35
  norm_num
  sorry

#eval number_of_distributions 8 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_drawers_l830_83032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l830_83013

theorem function_properties (ω φ : ℝ) : 
  ω > 0 → 
  |φ| ≤ π/2 → 
  (∀ x₁ x₂, x₁ < x₂ ∧ 2 * Real.sin (ω * x₁ + φ) + 1 = -1 ∧ 2 * Real.sin (ω * x₂ + φ) + 1 = -1 → x₂ - x₁ = π) →
  (∀ x, x ∈ Set.Ioo (-π/12) (π/3) → 2 * Real.sin (ω * x + φ) + 1 > 1) →
  φ ∈ Set.Icc (π/6) (π/3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l830_83013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_inequality_l830_83040

theorem plane_division_inequality (n : ℕ) (h : n ≥ 2) :
  (n * (n + 1) / 2 + 1 : ℕ) < n + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_inequality_l830_83040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_play_wins_majority_best_play_wins_majority_multi_l830_83097

/-- The probability of the best play winning in a two-play competition -/
noncomputable def best_play_wins_probability (n : ℕ) : ℝ :=
  1 - (1 / 2) ^ n

/-- Theorem: The probability of the best play winning with a majority of votes -/
theorem best_play_wins_majority (n : ℕ) :
  let total_mothers : ℕ := 2 * n
  let prob_vote_best : ℝ := 1 / 2
  let prob_vote_child : ℝ := 1 / 2
  best_play_wins_probability n = 1 - (1 / 2) ^ n := by
  sorry

/-- The probability of the best play winning in a multi-play competition -/
noncomputable def best_play_wins_probability_multi (n : ℕ) (s : ℕ) : ℝ :=
  1 - (1 / 2) ^ ((s - 1) * n)

/-- Theorem: The probability of the best play winning with a majority of votes in a multi-play competition -/
theorem best_play_wins_majority_multi (n : ℕ) (s : ℕ) :
  let total_mothers : ℕ := s * n
  let prob_vote_best : ℝ := 1 / 2
  let prob_vote_child : ℝ := 1 / 2
  best_play_wins_probability_multi n s = 1 - (1 / 2) ^ ((s - 1) * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_play_wins_majority_best_play_wins_majority_multi_l830_83097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_of_two_numbers_l830_83041

theorem hcf_of_two_numbers (a b : ℕ) (H : ℕ) : 
  (∃ (lcm : ℕ), lcm = H * 12 * 13 ∧ Nat.lcm a b = lcm) →
  299 ∈ ({a, b} : Set ℕ) →
  13 ∣ 299 →
  Nat.gcd a b = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_of_two_numbers_l830_83041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_polar_coordinates_l830_83076

/-- The polar coordinates of the intersection point of two curves -/
theorem intersection_point_polar_coordinates 
  (ρ : ℝ → ℝ) (θ : ℝ) 
  (h1 : ρ θ * (Real.cos θ + Real.sin θ) = 1) 
  (h2 : ρ θ * (Real.sin θ - Real.cos θ) = 1) 
  (h3 : 0 ≤ θ ∧ θ < 2 * Real.pi) : 
  ρ θ = 1 ∧ θ = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_polar_coordinates_l830_83076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cost_is_correct_l830_83063

noncomputable def shirt_prices : List ℚ := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
noncomputable def returned_prices : List ℚ := [20, 25, 30, 22, 23, 29]
def discount_rate : ℚ := 1/10
def tax_rate : ℚ := 2/25

noncomputable def final_cost (prices : List ℚ) (returned : List ℚ) (discount : ℚ) (tax : ℚ) : ℚ :=
  let kept_prices := prices.filter (fun p => p ∉ returned)
  let subtotal := kept_prices.sum
  let discounted_price := subtotal * (1 - discount)
  discounted_price * (1 + tax)

theorem final_cost_is_correct :
  final_cost shirt_prices returned_prices discount_rate tax_rate = 8262/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cost_is_correct_l830_83063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_three_from_six_l830_83007

def red_balls : Finset (Fin 3) := Finset.univ
def blue_balls : Finset (Fin 3) := Finset.univ
def all_balls : Finset (Fin 6) := Finset.univ

theorem select_three_from_six :
  (Finset.filter (fun s => s.card = 3) (Finset.powerset all_balls)).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_three_from_six_l830_83007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_pi_half_equals_cos_l830_83061

theorem sin_plus_pi_half_equals_cos (x : ℝ) : Real.sin (x + π / 2) = Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_pi_half_equals_cos_l830_83061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_even_iff_bases_even_and_sum_div_4_l830_83048

structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  height : ℝ
  is_isosceles : Prop
  AD_perpendicular_BC : Prop
  AD_tangent_to_circle : Prop
  T_is_midpoint_of_AD : Prop
  BC_passes_through_circle_center : Prop

noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  (t.AB + t.CD) * t.height / 2

theorem area_is_even_iff_bases_even_and_sum_div_4 (t : IsoscelesTrapezoid) :
  ∃ n : ℕ, area t = ↑(2 * n) ↔
  ∃ a b : ℕ, t.AB = ↑(2 * a) ∧ t.CD = ↑(2 * b) ∧ ∃ k : ℕ, a + b = 2 * k :=
by
  sorry

#check area_is_even_iff_bases_even_and_sum_div_4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_even_iff_bases_even_and_sum_div_4_l830_83048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l830_83093

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- For any four points A, B, C, D in a vector space, 
    BA + CB - CD + 2*AD = AD -/
theorem vector_equality (A B C D : V) :
  (B - A) + (C - B) - (C - D) + (2 : ℝ) • (A - D) = A - D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l830_83093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l830_83019

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 2 = 1

/-- Definition of the line l passing through the right focus -/
noncomputable def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 2

/-- Area of triangle OAB -/
noncomputable def area_OAB (m : ℝ) : ℝ :=
  2 * Real.sqrt 6 * Real.sqrt (m^2 + 1) / (m^2 + 3)

/-- Theorem stating the properties of ellipse C and line l -/
theorem ellipse_and_line_properties :
  (∀ x y, ellipse_C x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  (ellipse_C 2 (Real.sqrt 6 / 3)) ∧
  (∀ m, IsMaxOn area_OAB Set.univ m ↔ m = 1 ∨ m = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l830_83019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archibald_games_won_l830_83092

/-- Given that Archibald and his brother are playing tennis against each other,
    prove that Archibald has won 12 games. -/
theorem archibald_games_won (total_games : ℕ) (archibald_games : ℕ) : 
  (archibald_games : ℚ) / (total_games : ℚ) = 2 / 5 →
  total_games = archibald_games + 18 →
  archibald_games = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archibald_games_won_l830_83092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l830_83015

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/24 = 1

-- Define the foci
def F1 : ℝ × ℝ := (5, 0)
def F2 : ℝ × ℝ := (-5, 0)

-- Define a point P on the hyperbola in the first quadrant
noncomputable def P : ℝ × ℝ := sorry

-- Define the distance ratio condition
axiom distance_ratio : abs (P.1 - F1.1) / abs (P.1 - F2.1) = 4/3

-- State the theorem
theorem inscribed_circle_radius :
  hyperbola P.1 P.2 →
  P.1 > 0 →
  P.2 > 0 →
  let r := (abs (P.1 - F1.1) + abs (P.1 - F2.1) - abs (F1.1 - F2.1)) / 2
  r = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l830_83015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_99_of_1_eq_one_tenth_l830_83023

noncomputable def f (x : ℝ) : ℝ := x / Real.sqrt (1 + x^2)

noncomputable def f_iter : ℕ → ℝ → ℝ
| 0 => id
| n + 1 => f ∘ (f_iter n)

theorem f_99_of_1_eq_one_tenth :
  f_iter 99 1 = (1 : ℝ) / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_99_of_1_eq_one_tenth_l830_83023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cube_over_t_cube_gt_cos_l830_83094

theorem sin_cube_over_t_cube_gt_cos (t : ℝ) (h1 : 0 < t) (h2 : t ≤ π / 2) :
  (Real.sin t / t)^3 > Real.cos t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cube_over_t_cube_gt_cos_l830_83094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_solution_l830_83010

noncomputable def a : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => a (n + 1) + a n

noncomputable def x : ℕ → ℝ → ℝ
  | 0, _ => 0  -- Add a base case for 0
  | 1, x₁ => x₁
  | (n + 2), x₁ => (a n + a (n - 1) * x₁) / (a (n + 1) + a n * x₁)

theorem x_solution (x₁ : ℝ) (h : x 2004 x₁ = 1 / x₁ - 1) :
  x₁ = (-1 + Real.sqrt 5) / 2 ∨ x₁ = (-1 - Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_solution_l830_83010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_twelve_l830_83075

theorem fourth_root_sixteen_to_twelve : (16 : ℝ) ^ ((1/4 : ℝ) * 12) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_twelve_l830_83075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l830_83043

/-- An arithmetic sequence with first term 1 and common difference d -/
def arithmeticSequence (d : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => arithmeticSequence d n + d

theorem arithmetic_geometric_sequence (d : ℝ) (h : d ≠ 0) :
  (arithmeticSequence d 0) * (arithmeticSequence d 4) = (arithmeticSequence d 1)^2 → d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l830_83043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_rectangles_ratio_l830_83004

/-- Given a rectangle ABCD with length 2 and width 1, and a similar rectangle EFGH attached to side AB
    with a side ratio of 1:2 to ABCD, prove that EC = 1/3 -/
theorem similar_rectangles_ratio (A B C D E F G H : ℝ × ℝ) :
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let EF := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let EC := Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2)
  AB = 2 →
  AD = 1 →
  EF = AB / 2 →
  EC = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_rectangles_ratio_l830_83004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_in_special_triangle_l830_83036

theorem cos_A_in_special_triangle (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  (∃ (k : ℝ), k > 0 ∧ Real.sin A = k * Real.sqrt 2 ∧ Real.sin B = k ∧ Real.sin C = 2 * k) →
  Real.cos A = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_in_special_triangle_l830_83036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_half_speed_l830_83066

/-- Calculates the speed for the second half of a journey given the total distance,
    speed for the first half, and total travel time. -/
theorem second_half_speed (total_distance : ℝ) (first_half_speed : ℝ) (total_time : ℝ) :
  total_distance = 26.67 →
  first_half_speed = 5 →
  total_time = 6 →
  ∃ (second_half_speed : ℝ),
    (total_distance / 2) / first_half_speed + (total_distance / 2) / second_half_speed = total_time ∧
    abs (second_half_speed - 20) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_half_speed_l830_83066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l830_83058

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and points M and N on the hyperbola symmetric about the origin,
    and P a moving point on the hyperbola,
    with slopes of PM and PN as k₁ and k₂ respectively (k₁ · k₂ ≠ 0),
    if the minimum value of |k₁| + |k₂| is 1,
    then the eccentricity of the hyperbola is √5/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (M N P : ℝ × ℝ) (k₁ k₂ : ℝ) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → (x, y) = M ∨ (x, y) = N ∨ (x, y) = P) →
  (M.1 = -N.1 ∧ M.2 = -N.2) →
  k₁ * k₂ ≠ 0 →
  (∀ k₁' k₂', |k₁'| + |k₂'| ≥ |k₁| + |k₂|) →
  |k₁| + |k₂| = 1 →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 5 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l830_83058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_face_sums_for_1_to_6_exists_equal_face_sums_for_1_to_10_l830_83088

-- Define a tetrahedron
structure Tetrahedron :=
  (vertices : Fin 4 → ℕ)
  (edges : Fin 6 → ℕ)
  (edge_midpoints : Fin 6 → ℕ)

-- Define the face sum function
def face_sum (t : Tetrahedron) (face : Fin 4) : ℕ := sorry

-- Theorem for part (a)
theorem no_equal_face_sums_for_1_to_6 :
  ¬ ∃ (t : Tetrahedron), 
    (∀ e : Fin 6, t.edges e ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) ∧ 
    (∀ f₁ f₂ : Fin 4, face_sum t f₁ = face_sum t f₂) :=
sorry

-- Theorem for part (b)
theorem exists_equal_face_sums_for_1_to_10 :
  ∃ (t : Tetrahedron),
    (∀ v : Fin 4, t.vertices v ∈ Finset.range 10) ∧
    (∀ e : Fin 6, t.edge_midpoints e ∈ Finset.range 10) ∧
    (∀ f₁ f₂ : Fin 4, face_sum t f₁ = face_sum t f₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_face_sums_for_1_to_6_exists_equal_face_sums_for_1_to_10_l830_83088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l830_83003

/-- The minimum area of a triangle formed by two fixed points and a point on a given circle -/
theorem min_triangle_area (A B : ℝ × ℝ) (circle : ℝ × ℝ → Prop) : 
  A = (-2, 0) →
  B = (0, 2) →
  (∀ x y, circle (x, y) ↔ x^2 + y^2 - 2*x = 0) →
  (∃ min_area : ℝ, 
    min_area = 3 - Real.sqrt 2 ∧
    ∀ C, circle C → 
      (abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) ≥ min_area)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l830_83003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_knight_is_player_9_l830_83099

/-- Represents a player in the team -/
inductive Player : Type
| mk : Fin 11 → Player

/-- Represents whether a player is a knight or a liar -/
inductive PlayerType
| Knight
| Liar

/-- The claim made by each player -/
def player_claim (p : Player) : ℕ :=
  match p with
  | Player.mk n => n.val + 1

/-- The type of a given player -/
def player_type (p : Player) : PlayerType :=
  sorry

/-- The statement is true if the player is a knight and false if the player is a liar -/
def statement_truth (p : Player) : Prop :=
  match player_type p with
  | PlayerType.Knight => ∃ (k l : ℕ), k + l = 11 ∧ (k : Int) - l = player_claim p
  | PlayerType.Liar => ¬∃ (k l : ℕ), k + l = 11 ∧ (k : Int) - l = player_claim p

theorem unique_knight_is_player_9 :
  (∃! p : Player, player_type p = PlayerType.Knight) ∧
  (∃ p : Player, player_type p = PlayerType.Knight ∧ p = Player.mk 8) :=
by
  sorry

#check unique_knight_is_player_9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_knight_is_player_9_l830_83099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_solution_l830_83020

/-- The sum of the infinite series 1 + 3x + 5x^2 + 7x^3 + ... -/
noncomputable def T (x : ℝ) : ℝ := (1 + 2*x) / ((1 - x)^2)

/-- The theorem stating that if T(x) = 25, then x = (26 - √76) / 25 -/
theorem infinite_series_solution :
  ∀ x : ℝ, T x = 25 → x = (26 - Real.sqrt 76) / 25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_solution_l830_83020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trailing_zeros_l830_83087

/-- The number of trailing zeros in a natural number -/
def trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n.factors.filter (· = 2)).length.min (n.factors.filter (· = 5)).length

/-- Given three natural numbers that sum to 1003, the maximum number of trailing zeros in their product is 7 -/
theorem max_trailing_zeros (a b c : ℕ) (h_sum : a + b + c = 1003) : 
  (∀ x y z : ℕ, x + y + z = 1003 → trailing_zeros (x * y * z) ≤ trailing_zeros (a * b * c)) →
  trailing_zeros (a * b * c) = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trailing_zeros_l830_83087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_zero_l830_83053

/-- The function f(x) = 2 - x^2 + x^3 -/
def f (x : ℝ) : ℝ := 2 - x^2 + x^3

/-- The function g(x) = 2 + x^2 + x^3 -/
def g (x : ℝ) : ℝ := 2 + x^2 + x^3

/-- The set of x-coordinates of intersection points -/
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

/-- Theorem: The maximum difference between y-coordinates of intersection points is 0 -/
theorem max_difference_zero : 
  ∀ (x y : ℝ), x ∈ intersection_points → y ∈ intersection_points → |f x - f y| = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_zero_l830_83053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shape_l830_83027

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the set of points satisfying φ = c
def ConeSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

-- Define a predicate for cone surface
def IsConeSurface (s : Set SphericalCoord) : Prop :=
  ∃ (c : ℝ), ∀ (p : SphericalCoord), p ∈ s ↔ p.φ = c

-- Theorem statement
theorem cone_shape (c : ℝ) : 
  ∃ (cone : Set SphericalCoord), ConeSet c = cone ∧ IsConeSurface cone := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shape_l830_83027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_prime_l830_83021

open Nat

theorem consecutive_sum_prime (m : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ k : ℕ, (Finset.range m).sum (λ i => k + i) = p) → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_prime_l830_83021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_two_l830_83014

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  f : PolarPoint → ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  g : ℝ → ℝ

/-- The given line in polar coordinates -/
noncomputable def givenLine : PolarLine :=
  { f := fun p => p.ρ * Real.cos p.θ - Real.sqrt 3 * p.ρ * Real.sin p.θ - 1 }

/-- The given circle in polar coordinates -/
noncomputable def givenCircle : PolarCircle :=
  { g := fun θ => 2 * Real.cos θ }

/-- Theorem: The length of the chord formed by the intersection of the given line and circle is 2 -/
theorem chord_length_is_two (A B : PolarPoint)
  (hA : givenLine.f A = 0 ∧ A.ρ = givenCircle.g A.θ)
  (hB : givenLine.f B = 0 ∧ B.ρ = givenCircle.g B.θ)
  (hAB_distinct : A ≠ B) :
  Real.sqrt ((A.ρ * Real.cos A.θ - B.ρ * Real.cos B.θ)^2 +
             (A.ρ * Real.sin A.θ - B.ρ * Real.sin B.θ)^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_two_l830_83014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_four_l830_83074

/-- A rectangular prism with dimensions a, b, c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A pyramid formed by vertices A, B, C, G of a rectangular prism -/
structure Pyramid (prism : RectangularPrism) where

/-- The volume of a pyramid -/
noncomputable def pyramidVolume (prism : RectangularPrism) (p : Pyramid prism) : ℝ := 
  (1 / 3) * (prism.a * prism.b) * prism.c

theorem pyramid_volume_is_four (prism : RectangularPrism) 
  (h1 : prism.a = 2) 
  (h2 : prism.b = 3) 
  (h3 : prism.c = 4) 
  (p : Pyramid prism) : 
  pyramidVolume prism p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_four_l830_83074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l830_83070

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line
def line_eq (x y c : ℝ) : Prop := y = x + c

-- Define the triangle
def triangle (c : ℝ) : Set (ℝ × ℝ) :=
  {(0, 0)} ∪ {(x, y) | circle_eq x y ∧ line_eq x y c}

-- Define the area of the triangle
noncomputable def triangle_area (c : ℝ) : ℝ := 3 * Real.sqrt 2 * |c| / 4

-- Theorem statement
theorem triangle_area_bounds (c : ℝ) :
  (9 ≤ triangle_area c ∧ triangle_area c ≤ 36) ↔ c ∈ Set.Icc (-4 * Real.sqrt 2) (4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l830_83070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_expression_equals_19_l830_83035

theorem ceiling_expression_equals_19 : ⌈2 * (10 - 3/4 : ℚ)⌉ = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_expression_equals_19_l830_83035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l830_83064

theorem binomial_product : (Nat.choose 12 3) * (Nat.choose 9 3) = 18480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l830_83064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ghost_perimeter_l830_83054

/-- The perimeter of a circular sector with a missing segment --/
theorem ghost_perimeter (r : ℝ) (missing_angle : ℝ) : 
  r > 0 → 
  missing_angle > 0 → 
  missing_angle < 2 * Real.pi →
  let remaining_angle := 2 * Real.pi - missing_angle;
  let arc_length := (remaining_angle / (2 * Real.pi)) * (2 * Real.pi * r);
  let perimeter := arc_length + 2 * r;
  r = 2 ∧ missing_angle = Real.pi / 2 → perimeter = 3 * Real.pi + 4 := by
  sorry

#check ghost_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ghost_perimeter_l830_83054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_count_l830_83098

theorem trapezoid_bases_count : ∃! n : ℕ,
  n = (Finset.filter (λ p : ℕ × ℕ ↦
    10 ∣ p.1 ∧ 10 ∣ p.2 ∧ p.1 + p.2 = 100 ∧ p.1 ≤ p.2)
    (Finset.product (Finset.range 101) (Finset.range 101))).card ∧ n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_count_l830_83098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l830_83090

theorem solve_exponential_equation (x : ℝ) : (3 : ℝ)^(x - 4) = (9 : ℝ)^3 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l830_83090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_calculation_l830_83083

/-- Represents the scale of a map in terms of inches to feet -/
structure MapScale where
  inches : ℚ
  feet : ℚ

/-- Calculates the actual distance in feet given a map distance in inches and a map scale -/
def actual_distance (map_distance : ℚ) (scale : MapScale) : ℚ :=
  (map_distance * scale.feet) / scale.inches

/-- Theorem: A 7.25 inch line on a map with a scale of 2 inches to 500 feet represents 1812.5 feet -/
theorem map_distance_calculation (scale : MapScale) 
  (h1 : scale.inches = 2) 
  (h2 : scale.feet = 500) : 
  actual_distance (7 + 1/4) scale = 1812 + 1/2 := by
  sorry

#eval actual_distance (7 + 1/4) ⟨2, 500⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_calculation_l830_83083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_ball_volume_ratio_l830_83029

/-- A cylinder containing a ball that is tangent to the side, top, and bottom surfaces -/
structure TangentBallCylinder where
  R : ℝ

/-- The volume of the ball -/
noncomputable def TangentBallCylinder.ball_volume (c : TangentBallCylinder) : ℝ :=
  (4 / 3) * Real.pi * c.R^3

/-- The volume of the cylinder -/
noncomputable def TangentBallCylinder.cylinder_volume (c : TangentBallCylinder) : ℝ :=
  2 * Real.pi * c.R^3

/-- The ratio of the volume of the cylinder to the volume of the ball is 3/2 -/
theorem cylinder_ball_volume_ratio (c : TangentBallCylinder) :
  c.cylinder_volume / c.ball_volume = 3 / 2 := by
  -- Unfold the definitions
  unfold TangentBallCylinder.cylinder_volume
  unfold TangentBallCylinder.ball_volume
  -- Perform algebraic simplification
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_ball_volume_ratio_l830_83029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_proof_l830_83025

def gas_cost_calculation (initial_reading final_reading : ℕ)
                         (fuel_consumption : ℚ) (gas_price : ℚ) : ℚ :=
  let distance := final_reading - initial_reading
  let gallons_used := (distance : ℚ) / fuel_consumption
  gallons_used * gas_price

theorem gas_cost_proof (initial_reading : ℕ) (final_reading : ℕ)
                       (fuel_consumption : ℚ) (gas_price : ℚ)
                       (h1 : initial_reading = 85432)
                       (h2 : final_reading = 85470)
                       (h3 : fuel_consumption = 25)
                       (h4 : gas_price = 385/100) :
  (⌊gas_cost_calculation initial_reading final_reading fuel_consumption gas_price * 100 + 1/2⌋ : ℚ) / 100 = 585/100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_proof_l830_83025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l830_83018

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^x + (3 : ℝ)^x + (6 : ℝ)^x = (7 : ℝ)^x :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l830_83018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l830_83072

/-- The chord length cut by a circle from a line --/
theorem chord_length_circle_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 3 = 0}
  let center : ℝ × ℝ := (0, 0)
  let distance_center_to_line := |3| / Real.sqrt 2
  let chord_length := 2 * Real.sqrt (9 - distance_center_to_line^2)
  chord_length = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l830_83072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l830_83028

-- Define the propositions
def p : Prop := ∃ x₀, x₀ > -2 ∧ 6 + |x₀| = 5

def q : Prop := ∀ x < 0, x^2 + 4/x^2 ≥ 4

def r : Prop := ∀ a ≥ 1, Monotone (fun x : ℝ => a * x + Real.cos x)

-- Define the negation of r
def not_r : Prop := ∃ a < 1, ¬Monotone (fun x : ℝ => a * x + Real.cos x)

-- Theorem statement
theorem propositions_truth :
  (¬r ↔ not_r) ∧ ¬p ∧ (p ∨ r) ∧ ¬(p ∧ q) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l830_83028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_G_l830_83030

/-- The function F as defined in the problem -/
def F (p q : ℝ) : ℝ := -2*p*q + 3*p*(1-q) + 3*(1-p)*q - 4*(1-p)*(1-q)

/-- The function G as defined in the problem -/
noncomputable def G (p : ℝ) : ℝ := 
  ⨆ q ∈ Set.Icc 0 1, F p q

theorem minimize_G :
  ∃ p, 0 ≤ p ∧ p ≤ 1 ∧ p = 7/12 ∧ ∀ p', 0 ≤ p' ∧ p' ≤ 1 → G p ≤ G p' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_G_l830_83030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_age_l830_83000

theorem sandy_age (sandy_age molly_age : ℚ) 
  (h1 : molly_age = sandy_age + 20) 
  (h2 : sandy_age / molly_age = 7 / 9) : 
  sandy_age = 70 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_age_l830_83000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l830_83031

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- State the theorem
theorem f_derivative (x : ℝ) (hx : x ≠ 0) : 
  deriv f x = 1 - 1/(x^2) := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l830_83031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_forms_spherical_surfaces_l830_83081

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Represents the given data for triangle construction -/
structure TriangleData where
  A : Point2D
  angle_diff : ℝ  -- (β - γ)
  f_a : ℝ         -- length of angle bisector
  ratio : ℝ       -- (b + c) : a

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point2D
  radius : ℝ

/-- Checks if a triangle satisfies the construction conditions -/
def satisfies_construction (t : Triangle) (data : TriangleData) : Prop := sorry

/-- Checks if a point is contained in a sphere -/
def Sphere.contains (s : Sphere) (p : Point2D) : Prop := 
  (p.x - s.center.x) ^ 2 + (p.y - s.center.y) ^ 2 ≤ s.radius ^ 2

/-- 
Given the data for triangle construction, proves that the locus of points B and C 
forms spherical surfaces when the endpoint D of the angle bisector is on the first projection plane.
-/
theorem locus_forms_spherical_surfaces (data : TriangleData) : 
  ∃ (s1 s2 : Sphere), 
    (∀ (t : Triangle), 
      satisfies_construction t data → 
      (Sphere.contains s1 t.B ∧ Sphere.contains s2 t.C)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_forms_spherical_surfaces_l830_83081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gloria_dimes_count_l830_83034

theorem gloria_dimes_count (quarters dimes : ℕ) : 
  dimes = 5 * quarters ∧
  (3 * quarters + 5 * dimes : ℚ) = 392 * 5 →
  dimes = 350 := by
  sorry

#check gloria_dimes_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gloria_dimes_count_l830_83034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l830_83008

/-- The distance from a point on the parabola y^2 = -4x with x-coordinate -6 to its focus is 7 -/
theorem parabola_focus_distance : ∀ y : ℝ, y^2 = -4*(-6) → 
  ∃ F : ℝ × ℝ, Real.sqrt (((-6 - F.1)^2 + (y - F.2)^2) : ℝ) = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l830_83008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_127_equals_binary_1010111_l830_83067

/-- Converts an octal digit to its binary representation -/
def octal_to_binary (d : Nat) : List Bool :=
  match d with
  | 0 => [false, false, false]
  | 1 => [false, false, true]
  | 2 => [false, true, false]
  | 3 => [false, true, true]
  | 4 => [true, false, false]
  | 5 => [true, false, true]
  | 6 => [true, true, false]
  | 7 => [true, true, true]
  | _ => []

/-- Converts an octal number to its binary representation -/
def octal_to_binary_number (octal : List Nat) : List Bool :=
  octal.bind octal_to_binary

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

theorem octal_127_equals_binary_1010111 :
  binary_to_decimal (octal_to_binary_number [1, 2, 7]) = 
  binary_to_decimal [true, false, true, false, true, true, true] := by
  sorry

#eval octal_to_binary_number [1, 2, 7]
#eval binary_to_decimal [true, false, true, false, true, true, true]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_127_equals_binary_1010111_l830_83067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_w_l830_83050

-- Define the complex number z
def z (a : ℝ) : ℂ := (a - 1) + Complex.I

-- Define the condition that z is purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

-- Define the complex number we're interested in
noncomputable def w (a : ℝ) : ℂ := (2 + Real.sqrt 2 * Complex.I) / (a - Complex.I)

-- State the theorem
theorem modulus_of_w (a : ℝ) (h : is_purely_imaginary (z a)) : 
  Complex.abs (w a) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_w_l830_83050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_a_eq_count_b_l830_83039

/-- A function with periodic properties -/
def PeriodicFunction (n m : ℕ+) :=
  {f : ℤ × ℤ → Fin 2 // ∀ i j : ℤ, f (i, j) = f (i + n, j) ∧ f (i, j) = f (i, j + m)}

/-- The set [k] = {1, 2, ..., k} -/
def range (k : ℕ+) : Finset ℕ := Finset.range k.val

/-- Count of (i, j) satisfying f(i, j) = f(i+1, j) = f(i, j+1) -/
def count_a (n m : ℕ+) (f : PeriodicFunction n m) : ℕ :=
  (range n).product (range m) |>.filter (fun p =>
    f.val (p.1, p.2) = f.val (p.1 + 1, p.2) ∧ f.val (p.1, p.2) = f.val (p.1, p.2 + 1)
  ) |>.card

/-- Count of (i, j) satisfying f(i, j) = f(i-1, j) = f(i, j-1) -/
def count_b (n m : ℕ+) (f : PeriodicFunction n m) : ℕ :=
  (range n).product (range m) |>.filter (fun p =>
    f.val (p.1, p.2) = f.val (p.1 - 1, p.2) ∧ f.val (p.1, p.2) = f.val (p.1, p.2 - 1)
  ) |>.card

/-- The main theorem: count_a equals count_b for any periodic function -/
theorem count_a_eq_count_b (n m : ℕ+) (f : PeriodicFunction n m) :
  count_a n m f = count_b n m f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_a_eq_count_b_l830_83039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l830_83047

-- Define the variables
variable (x₁ y₁ x₂ y₂ : ℤ)

-- Define the points A and B
def A : ℤ × ℤ := (x₁, y₁)
def B : ℤ × ℤ := (x₂, y₂)

-- Define the conditions
axiom positive_coords : x₁ > 0 ∧ y₁ > 0 ∧ x₂ > 0 ∧ y₂ > 0
axiom angle_OA : y₁ > x₁
axiom angle_OB : x₂ > y₂
axiom area_difference : x₂ * y₂ = x₁ * y₁ + 67

-- Theorem to prove
theorem unique_solution : x₁ = 1 ∧ y₁ = 5 ∧ x₂ = 9 ∧ y₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l830_83047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_false_proposition_l830_83052

-- Define the function f(x) = ln x - (1/2)x^2
noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2) * x^2

-- State the theorem
theorem range_of_a_for_false_proposition :
  {a : ℝ | ∃ x > 0, f x ≥ a} = Set.Iic (-1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_false_proposition_l830_83052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_rental_cost_per_sqft_l830_83055

/-- Represents the square footage of a house with various rooms -/
structure HouseSquareFootage where
  masterBedroomBath : ℝ
  guestBedroom : ℝ
  otherAreas : ℝ

/-- Calculates the total square footage of the house -/
noncomputable def totalSquareFootage (h : HouseSquareFootage) : ℝ :=
  h.masterBedroomBath + 2 * h.guestBedroom + h.otherAreas

/-- Calculates the cost per square foot given the total rent and house square footage -/
noncomputable def costPerSquareFoot (totalRent : ℝ) (h : HouseSquareFootage) : ℝ :=
  totalRent / totalSquareFootage h

/-- Theorem statement for Tony's rental cost per square foot -/
theorem tony_rental_cost_per_sqft :
  let h : HouseSquareFootage := {
    masterBedroomBath := 500,
    guestBedroom := 200,
    otherAreas := 600
  }
  costPerSquareFoot 3000 h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_rental_cost_per_sqft_l830_83055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_b_negative_l830_83069

theorem a_positive_b_negative (m : ℝ) (a b : ℝ) 
  (h1 : (9 : ℝ)^m = 10) 
  (h2 : a = (10 : ℝ)^m - 11) 
  (h3 : b = (8 : ℝ)^m - 9) : 
  a > 0 ∧ 0 > b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_b_negative_l830_83069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l830_83068

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * (n : ℚ) / 2

theorem fourth_term_value (seq : ArithmeticSequence) 
  (sum_7 : sum_n seq 7 = 35) : 
  seq.a 4 = 5 := by
  sorry

#check fourth_term_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l830_83068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_theorem_l830_83022

theorem angle_terminal_side_theorem (α : ℝ) (h : ∃ (t : ℝ), t > 0 ∧ t * (Real.cos α) = -5 ∧ t * (Real.sin α) = 12) :
  Real.sin (-π - α) - 2 * Real.cos (π - α) = 22 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_theorem_l830_83022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_M_N_l830_83085

-- Define the set difference
def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define the symmetric difference
def symmetric_difference (M N : Set ℝ) : Set ℝ := (set_difference M N) ∪ (set_difference N M)

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4*x + 3}

-- Define set N
def N : Set ℝ := {y | ∃ x : ℝ, y = -2^x}

-- Theorem statement
theorem symmetric_difference_M_N : 
  symmetric_difference M N = Set.Ioi (-1) ∪ Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_M_N_l830_83085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suresh_work_hours_l830_83042

theorem suresh_work_hours (suresh_rate ashutosh_rate : ℚ) 
  (ashutosh_remaining_time : ℚ) : ℚ :=
  let x : ℚ := 3
  have h1 : suresh_rate = 1 / 15 := by sorry
  have h2 : ashutosh_rate = 1 / 30 := by sorry
  have h3 : ashutosh_remaining_time = 12 := by sorry
  have h4 : x * suresh_rate + ashutosh_remaining_time * ashutosh_rate = 1 := by sorry
  x

#check suresh_work_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suresh_work_hours_l830_83042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_row_is_3122_l830_83082

/-- Represents a cell in the 3x3 grid -/
structure Cell where
  value : ℕ
  deriving Repr

/-- Represents the 3x3 grid -/
def Grid := Matrix (Fin 3) (Fin 3) Cell

/-- Checks if the grid satisfies the arrow conditions -/
def satisfies_arrow_conditions (g : Grid) : Prop :=
  -- Add conditions for each arrow here
  true

/-- The four-digit number formed by the second row -/
def second_row_number (g : Grid) : ℕ :=
  1000 * (g 1 0).value + 100 * (g 1 1).value + 10 * (g 1 2).value + (g 1 3).value

/-- The main theorem stating that the second row forms 3122 -/
theorem second_row_is_3122 (g : Grid) (h : satisfies_arrow_conditions g) : 
  second_row_number g = 3122 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_row_is_3122_l830_83082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_circular_cone_central_angle_l830_83080

/-- Represents a right circular cone -/
structure RightCircularCone where
  /-- Radius of the base circle -/
  radius : ℝ
  /-- Slant height of the cone -/
  slant_height : ℝ
  /-- The axis section is an isosceles right triangle -/
  isosceles_right : slant_height = Real.sqrt 2 * radius

/-- The central angle of the unfolded side of a right circular cone -/
noncomputable def central_angle (cone : RightCircularCone) : ℝ :=
  (2 * Real.pi * cone.radius) / cone.slant_height

/-- Theorem: The central angle of the unfolded side of a right circular cone is √2π -/
theorem right_circular_cone_central_angle (cone : RightCircularCone) :
  central_angle cone = Real.sqrt 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_circular_cone_central_angle_l830_83080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l830_83086

noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 - 4*x + 4) / (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.univ \ {2, 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l830_83086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l830_83006

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l830_83006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_parabola_l830_83060

/-- The directrix of the parabola y = -3x^2 + 6x - 5 -/
noncomputable def directrix_parabola : ℝ := -25/12

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -3*x^2 + 6*x - 5

theorem directrix_of_parabola :
  ∃ (h k a : ℝ), a ≠ 0 ∧
  (∀ x, parabola x = a * (x - h)^2 + k) ∧
  directrix_parabola = k - 1 / (4 * a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_parabola_l830_83060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pebbles_for_60_boxes_alice_can_always_prevent_bob_from_winning_l830_83078

/-- The number of boxes in the game -/
def N : ℕ := 60

/-- The minimum number of pebbles Alice needs to prevent Bob from winning -/
def min_pebbles (N : ℕ) : ℕ := ((N / 2 + 1) + 1) * ((N / 2 + 1) + 1) - 1

/-- Theorem stating the minimum number of pebbles for 60 boxes -/
theorem min_pebbles_for_60_boxes :
  min_pebbles N = 960 :=
by sorry

/-- Predicate stating that no box is ever empty given Alice's and Bob's strategies -/
def no_box_empty (alice_strategy bob_strategy : List (Fin N → ℕ)) : Prop :=
  ∀ (round : ℕ) (box : Fin N), round < alice_strategy.length → 
    (alice_strategy.get ⟨round, by sorry⟩) box > 0

/-- Theorem stating that this number of pebbles is sufficient for Alice to always prevent Bob from winning -/
theorem alice_can_always_prevent_bob_from_winning (n : ℕ) (h : n ≥ min_pebbles N) :
  ∀ (strategy : List (Fin N → ℕ)), ∃ (alice_strategy : List (Fin N → ℕ)),
    no_box_empty alice_strategy strategy :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pebbles_for_60_boxes_alice_can_always_prevent_bob_from_winning_l830_83078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yolanda_walking_rate_l830_83079

/-- Calculates Yolanda's walking rate given the conditions of the problem -/
noncomputable def yolandasRate (totalDistance : ℝ) (bobsDelay : ℝ) (bobsRate : ℝ) (bobsDistance : ℝ) : ℝ :=
  let timeForBob := bobsDistance / bobsRate
  let timeForYolanda := timeForBob + bobsDelay
  let yolandasDistance := totalDistance - bobsDistance
  yolandasDistance / timeForYolanda

/-- Theorem stating that Yolanda's walking rate is 5 miles per hour given the problem conditions -/
theorem yolanda_walking_rate :
  yolandasRate 65 1 7 35 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yolanda_walking_rate_l830_83079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_mpg_calculation_l830_83046

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℝ
  city_miles_per_tankful : ℝ
  city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data -/
noncomputable def calculate_city_mpg (car : CarFuelEfficiency) : ℝ :=
  let highway_mpg := car.highway_miles_per_tankful / (car.highway_miles_per_tankful / (car.city_miles_per_tankful / (car.highway_miles_per_tankful / car.city_miles_per_tankful - car.city_mpg_difference)))
  highway_mpg - car.city_mpg_difference

/-- Theorem stating that the calculated city mpg is approximately 11.05 for the given car -/
theorem city_mpg_calculation (car : CarFuelEfficiency)
  (h1 : car.highway_miles_per_tankful = 720)
  (h2 : car.city_miles_per_tankful = 378)
  (h3 : car.city_mpg_difference = 10) :
  |calculate_city_mpg car - 11.05| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_mpg_calculation_l830_83046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_black_stone_count_l830_83026

/-- Represents the state of a box in the game -/
inductive BoxState
| Empty
| OneBlack
| Other

/-- Represents the game board as a 3x3 grid -/
def GameBoard := Fin 3 → Fin 3 → BoxState

/-- Counts the number of boxes with exactly one black stone -/
def countOneBlack (board : GameBoard) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 3)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 3)) fun j =>
      match board i j with
      | BoxState.OneBlack => 1
      | _ => 0

/-- Checks if a given board state is valid according to the game rules -/
def isValidBoard (board : GameBoard) : Prop :=
  -- Add specific conditions here based on the game rules
  sorry

/-- The main theorem stating the possible values for the number of boxes with one black stone -/
theorem one_black_stone_count (board : GameBoard) :
  isValidBoard board →
  let count := countOneBlack board
  count = 3 ∨ count = 4 ∨ count = 5 ∨ count = 6 ∨ count = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_black_stone_count_l830_83026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_relations_l830_83037

theorem exponential_relations (a b c : ℕ) 
  (ha : (2 : ℝ)^a = 3) 
  (hb : (2 : ℝ)^b = 6) 
  (hc : (2 : ℝ)^c = 12) : 
  ((2 : ℝ)^(c-b) = 2) ∧ (a + c = 2*b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_relations_l830_83037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l830_83051

/-- Represents the time taken to complete the work -/
noncomputable def D : ℝ := sorry

/-- Represents the total amount of work -/
noncomputable def W : ℝ := sorry

/-- a can complete the work in 24 days -/
axiom a_rate : W / 24 > 0

/-- b can complete the work in 30 days -/
axiom b_rate : W / 30 > 0

/-- c can complete the work in 40 days -/
axiom c_rate : W / 40 > 0

/-- The combined rate of a, b, and c working together -/
noncomputable def combined_rate : ℝ := W / 24 + W / 30 + W / 40

/-- The combined rate of a and b working together -/
noncomputable def ab_rate : ℝ := W / 24 + W / 30

/-- c left 4 days before the completion of the work -/
axiom c_left_early : D > 4

/-- The work is completed when the sum of work done equals the total work -/
axiom work_completed : (D - 4) * combined_rate + 4 * ab_rate = W

/-- The theorem to prove -/
theorem work_completion_time : D = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l830_83051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_target_height_l830_83056

noncomputable def initial_height : ℝ := 800
noncomputable def bounce_ratio : ℝ := 2/3
noncomputable def target_height : ℝ := 2

noncomputable def height_after_bounces (k : ℕ) : ℝ := initial_height * (bounce_ratio ^ k)

theorem min_bounces_to_target_height :
  ∀ k : ℕ, k < 17 → height_after_bounces k ≥ target_height ∧
  height_after_bounces 17 < target_height :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_target_height_l830_83056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bisecting_line_sum_l830_83009

noncomputable section

/-- Triangle PQR with vertices P(1, 6), Q(3, -2), and R(9, -2) -/
def triangle_PQR : Set (ℝ × ℝ) :=
  {(1, 6), (3, -2), (9, -2)}

/-- A line through a point (x, y) with slope m and y-intercept b -/
def line (x y m b : ℝ) : Set (ℝ × ℝ) :=
  {(a, c) | c = m * (a - x) + y ∧ c = m * a + b}

/-- The area of a triangle given by three points -/
def triangle_area (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * abs ((x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)))

/-- Theorem: For triangle PQR, if a line through R cuts the area in half,
    then the sum of its slope and y-intercept is 18/7 -/
theorem area_bisecting_line_sum (m b : ℝ) :
  (9, -2) ∈ line 9 (-2) m b →
  (∃ (x : ℝ), (x, m * x + b) ∈ triangle_PQR) →
  triangle_area 1 6 3 (-2) x (m * x + b) = (1/2) * triangle_area 1 6 3 (-2) 9 (-2) →
  m + b = 18/7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bisecting_line_sum_l830_83009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_15pi_4_l830_83084

noncomputable def f (x : ℝ) : ℝ :=
  if -Real.pi/2 ≤ x ∧ x ≤ 0 then Real.cos x
  else if 0 ≤ x ∧ x ≤ Real.pi then Real.sin x
  else 0  -- default case for completeness

theorem f_value_at_negative_15pi_4 (h1 : ∀ x, f (x + 3*Real.pi/2) = f x) 
  (h2 : ∀ x, -Real.pi/2 ≤ x → x ≤ 0 → f x = Real.cos x)
  (h3 : ∀ x, 0 ≤ x → x ≤ Real.pi → f x = Real.sin x) :
  f (-15*Real.pi/4) = Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_15pi_4_l830_83084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_with_property_l830_83096

-- Define the set M
def M : Finset ℕ := Finset.range 20

-- Define the function f
def f : {S : Finset ℕ // S ⊆ M ∧ S.card = 9} → ℕ := sorry

-- Main theorem
theorem exists_subset_with_property :
  ∃ T : Finset ℕ, T ⊆ M ∧ T.card = 10 ∧
  ∀ k ∈ T, f ⟨T.erase k, by sorry⟩ ≠ k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_with_property_l830_83096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l830_83059

-- Define the sets M and N
def M : Set ℝ := {x | |x - 4| + |x - 1| < 5}
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x - 6) < 0}

-- State the theorem
theorem intersection_sum (a b : ℝ) : 
  (M ∩ N a = Set.Ioo 2 b) → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l830_83059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squares_of_roots_max_sum_of_squares_of_roots_is_18_l830_83045

theorem max_sum_of_squares_of_roots : 
  ∃ k : ℝ, ∀ x₁ x₂ : ℝ, 
    x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0 ∧
    x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0 →
    x₁^2 + x₂^2 ≤ 18 := by
  sorry

theorem max_sum_of_squares_of_roots_is_18 :
  ∃ k x₁ x₂ : ℝ,
    x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0 ∧
    x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0 ∧
    x₁^2 + x₂^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squares_of_roots_max_sum_of_squares_of_roots_is_18_l830_83045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_tangents_l830_83002

/-- Two circles with radii in ratio 2:3, touching internally --/
structure InternallyTouchingCircles where
  k : ℝ
  R₁ : ℝ := 2 * k
  R₂ : ℝ := 3 * k

/-- Point on the larger circle where the perpendicular line intersects --/
noncomputable def IntersectionPoint (c : InternallyTouchingCircles) : ℝ × ℝ :=
  (c.k * Real.sqrt 5, 0)

/-- Tangent point on the smaller circle --/
def TangentPoint (c : InternallyTouchingCircles) : ℝ × ℝ :=
  (2 * c.k, 0)

/-- Theorem: The angle between tangents is 90 degrees --/
theorem angle_between_tangents (c : InternallyTouchingCircles) :
  let B := IntersectionPoint c
  let M := TangentPoint c
  Real.arccos ((B.1 - M.1) / (B.1 - c.R₁)) * (180 / Real.pi) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_tangents_l830_83002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_5x_squared_l830_83073

/-- Represents a trapezoid with height x and bases 4x and 2x -/
structure Trapezoid (x : ℝ) where
  height : ℝ := x
  long_base : ℝ := 4 * x
  short_base : ℝ := 2 * x

/-- The area of a trapezoid given its height and two bases -/
noncomputable def trapezoid_area (h b1 b2 : ℝ) : ℝ := (b1 + b2) / 2 * h

/-- Theorem stating that the area of the described trapezoid is 5x^2 -/
theorem trapezoid_area_is_5x_squared (x : ℝ) (trap : Trapezoid x) :
  trapezoid_area trap.height trap.long_base trap.short_base = 5 * x^2 := by
  sorry

#check trapezoid_area_is_5x_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_5x_squared_l830_83073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l830_83024

theorem puzzle_solution (x y z w : ℕ) 
  (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) (w_pos : w > 0)
  (h1 : x^3 = y^2) 
  (h2 : z^5 = w^4) 
  (h3 : z = x + 31) : 
  w = y + 2351 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l830_83024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_l830_83089

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + 4*x - a

-- State the theorem
theorem no_extreme_points (a : ℝ) : 
  ∀ x : ℝ, deriv (f a) x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_l830_83089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_determines_a_l830_83095

def U (a : ℝ) : Set ℝ := {-1, 2, 3, a}
def M : Set ℝ := {-1, 3}

theorem complement_determines_a (a : ℝ) (h : Set.compl M = {2, 5}) : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_determines_a_l830_83095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_difference_l830_83038

theorem parallel_vectors_tan_difference (α : ℝ) :
  let a : Fin 2 → ℝ := ![Real.cos α, -2]
  let b : Fin 2 → ℝ := ![Real.sin α, 1]
  (∃ (k : ℝ), a = k • b) →
  Real.tan (α - π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_difference_l830_83038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l830_83091

/-- The area of a right triangle with base 30 cm and height 40 cm is 600 cm². -/
theorem right_triangle_area (base height : Real)
  (h1 : base = 30)
  (h2 : height = 40) :
  (1 / 2) * base * height = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l830_83091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l830_83012

noncomputable section

/-- The number of hours on a clock face -/
def hours : ℕ := 12

/-- The number of minutes in an hour -/
def minutes : ℕ := 60

/-- The angle in degrees that the hour hand moves per hour -/
def hourHandDegPerHour : ℝ := 360 / hours

/-- The angle in degrees that the minute hand moves per minute -/
def minuteHandDegPerMinute : ℝ := 360 / minutes

/-- The position of the hour hand at a given hour and minute -/
def hourHandPosition (h : ℕ) (m : ℕ) : ℝ :=
  (h % hours : ℝ) * hourHandDegPerHour + (m : ℝ) * hourHandDegPerHour / minutes

/-- The position of the minute hand at a given minute -/
def minuteHandPosition (m : ℕ) : ℝ := (m : ℝ) * minuteHandDegPerMinute

/-- The smaller angle between two angles on a clock face -/
def smallerAngle (a b : ℝ) : ℝ :=
  min (abs (a - b)) (360 - abs (a - b))

theorem clock_angle_at_3_15 :
  smallerAngle (hourHandPosition 3 15) (minuteHandPosition 15) = 7.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l830_83012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_cube_configurations_l830_83077

/-- Represents a position where an additional square can be attached -/
inductive AttachmentPosition
| SideA
| SideB
| BottomD
| AdjacentD1
| AdjacentD2
| OppositeD
| Other

/-- Represents a square -/
structure Square :=
  (side : ℝ)

/-- Congruence relation for squares -/
def Square.congruent (s1 s2 : Square) : Prop :=
  s1.side = s2.side

/-- Represents the L-shaped polygon -/
structure LShape :=
  (squares : Fin 4 → Square)
  (isCongruent : ∀ i j, Square.congruent (squares i) (squares j))
  (isLShaped : Prop) -- Placeholder for the L-shape configuration

/-- Represents a configuration with an additional square attached -/
structure Configuration :=
  (base : LShape)
  (attachmentPosition : AttachmentPosition)

/-- Predicate to check if a configuration can form a cube missing one face -/
def canFormCubeMissingOneFace (config : Configuration) : Prop :=
  sorry -- Definition of this predicate

/-- The main theorem -/
theorem valid_cube_configurations (positions : Fin 7 → AttachmentPosition) :
  ∃ (validConfigs : Finset Configuration),
    validConfigs.card = 3 ∧
    (∀ config ∈ validConfigs, canFormCubeMissingOneFace config) ∧
    (∀ config : Configuration, 
      canFormCubeMissingOneFace config → config ∈ validConfigs) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_cube_configurations_l830_83077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_equals_neg_cos_l830_83065

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 0  -- Define for 0 to avoid missing case
  | 1 => Real.cos
  | n+1 => deriv (f n)

-- State the theorem
theorem f_2015_equals_neg_cos : f 2015 = λ x => -Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_equals_neg_cos_l830_83065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l830_83033

/-- Inverse proportion function -/
noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (m + 3) / x

/-- Predicate for a point being in the second or fourth quadrant -/
def in_second_or_fourth_quadrant (x y : ℝ) : Prop :=
  (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)

/-- Main theorem -/
theorem inverse_proportion_quadrants (m : ℝ) :
  (∀ x ≠ 0, in_second_or_fourth_quadrant x (inverse_proportion m x)) →
  m < -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l830_83033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_3750_l830_83011

theorem largest_prime_factor_of_3750 :
  (Nat.factors 3750).maximum? = some 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_3750_l830_83011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_primes_with_difference_divisible_by_four_l830_83044

theorem max_primes_with_difference_divisible_by_four :
  ∃ (S : Finset ℕ),
    (∀ n, n ∈ S → n ≤ 25 ∧ n ≥ 1) ∧
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → (a - b) % 4 = 0) ∧
    (∀ n, n ∈ S → Nat.Prime n) ∧
    S.card = 5 ∧
    (∀ T : Finset ℕ,
      (∀ n, n ∈ T → n ≤ 25 ∧ n ≥ 1) →
      (∀ a b, a ∈ T → b ∈ T → a ≠ b → (a - b) % 4 = 0) →
      (∀ n, n ∈ T → Nat.Prime n) →
      T.card ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_primes_with_difference_divisible_by_four_l830_83044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l830_83049

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line equation -/
def line (x y b : ℝ) : Prop := y = -1/2 * x + b

/-- Circle equation -/
def circle_eq (x y : ℝ) : Prop := (x - 24/5)^2 + (y + 4)^2 = 16

/-- Triangle area -/
noncomputable def triangleArea (b : ℝ) : ℝ := 4 * Real.sqrt 2 * Real.sqrt (b^3 + 2*b^2)

theorem parabola_line_intersection
  (b : ℝ)
  (h1 : ∃ (x1 y1 x2 y2 : ℝ), parabola x1 y1 ∧ parabola x2 y2 ∧ line x1 y1 b ∧ line x2 y2 b)
  (h2 : b < 0) :
  (∀ (x y : ℝ), circle_eq x y) ∧
  (∃ (b_max : ℝ), b_max = -4/3 ∧ triangleArea b_max = 32 * Real.sqrt 3 / 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l830_83049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_theorem_l830_83062

/-- Represents a 3-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if a ThreeDigitNumber has all different non-zero digits -/
def has_different_nonzero_digits (n : ThreeDigitNumber) : Prop :=
  let (x, y, z) := n
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  let (x, y, z) := n
  100 * x + 10 * y + z

/-- Checks if a number satisfies the product conditions -/
def satisfies_product_conditions (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = 6 ∧
  digits[0]? = digits[5]? ∧
  (∀ i ∈ [1, 2, 3, 4], digits[0]? ≠ digits[i]?) ∧
  ({digits[1]?, digits[2]?, digits[3]?, digits[4]?} : Set (Option Nat)) =
    ({digits[0]?, digits[1]?, digits[2]?, digits[5]?} : Set (Option Nat))

/-- The main theorem -/
theorem product_theorem :
  ∀ x y z : Nat,
  let n1 : ThreeDigitNumber := (x, y, z)
  let n2 : ThreeDigitNumber := (y, x, z)
  has_different_nonzero_digits n1 →
  satisfies_product_conditions (to_nat n1 * to_nat n2) →
  ((x = 2 ∧ y = 6 ∧ z = 9) ∨ (x = 3 ∧ y = 5 ∧ z = 9)) :=
by sorry

#check product_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_theorem_l830_83062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l830_83071

/-- Line L in parametric form -/
noncomputable def line_L (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 + (1/2) * t, (Real.sqrt 3 / 2) * t)

/-- Curve C in polar form -/
def curve_C : ℝ := 4

/-- The length of segment AB, where A and B are intersection points of line L and curve C -/
noncomputable def length_AB : ℝ := Real.sqrt 55

/-- Theorem stating that the length of segment AB is √55 -/
theorem length_of_segment_AB :
  ∃ t₁ t₂,
    let A := line_L t₁
    let B := line_L t₂
    (A.1^2 + A.2^2 = curve_C^2) ∧
    (B.1^2 + B.2^2 = curve_C^2) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = length_AB^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l830_83071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_height_is_two_l830_83057

/-- Regular pyramid with apex S and base ABCD -/
structure RegularPyramid where
  SA : ℝ
  base_side : ℝ

/-- Volume of a regular pyramid -/
noncomputable def volume (p : RegularPyramid) : ℝ :=
  (1/3) * p.base_side^2 * Real.sqrt (p.SA^2 - (Real.sqrt 2 * p.base_side / 2)^2)

/-- The height of the pyramid when its volume is maximized -/
noncomputable def max_volume_height (p : RegularPyramid) : ℝ :=
  Real.sqrt (p.SA^2 - (Real.sqrt 2 * p.base_side / 2)^2)

/-- Theorem stating that the height of the pyramid is 2 when its volume is maximized -/
theorem max_volume_height_is_two (p : RegularPyramid) (h : p.SA = 2 * Real.sqrt 3) :
  ∃ (a : ℝ), volume p = volume {SA := p.SA, base_side := a} ∧ 
             max_volume_height {SA := p.SA, base_side := a} = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_height_is_two_l830_83057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_sphere_radius_l830_83001

/-- Pyramid TNRSC with specific properties -/
structure Pyramid where
  -- Base is a convex quadrilateral TNRS
  baseIsConvexQuadrilateral : Prop
  -- Diagonal NS divides TNRS into two triangles of equal area
  diagonalDividesBaseEqually : Prop
  -- Length of edge TH
  TH : ℝ
  TH_eq : TH = 4
  -- Cotangent of angle HCP
  ctgHCP : ℝ
  ctgHCP_eq : ctgHCP = Real.sqrt 2
  -- Sum of edge lengths TK and CK
  TK_plus_CK : ℝ
  TK_plus_CK_eq : TK_plus_CK = 4
  -- Volume of the pyramid
  volume : ℝ
  volume_eq : volume = 16 / 3

/-- The radius of the largest sphere that can fit inside the pyramid TNRSC -/
noncomputable def largestInscribedSphereRadius (p : Pyramid) : ℝ := 2 - Real.sqrt 2

/-- Theorem stating that the radius of the largest inscribed sphere in the given pyramid is 2 - √2 -/
theorem largest_inscribed_sphere_radius (p : Pyramid) :
  largestInscribedSphereRadius p = 2 - Real.sqrt 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_sphere_radius_l830_83001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_l830_83016

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_conditions (a b : ℝ) :
  (∀ x ≠ 1, f a b (-x) = -f a b x) → a = -1/2 ∧ b = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_l830_83016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_one_subject_l830_83005

theorem students_in_one_subject
  (both geometry only_cs : ℕ)
  (h1 : both = 15)
  (h2 : geometry = 35)
  (h3 : only_cs = 18) :
  (geometry - both) + only_cs = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_one_subject_l830_83005
