import Mathlib

namespace NUMINAMATH_CALUDE_probability_other_note_counterfeit_l2266_226625

/-- Represents the total number of banknotes -/
def total_notes : ℕ := 20

/-- Represents the number of counterfeit notes -/
def counterfeit_notes : ℕ := 5

/-- Represents the number of genuine notes -/
def genuine_notes : ℕ := total_notes - counterfeit_notes

/-- Calculates the probability that both drawn notes are counterfeit -/
def prob_both_counterfeit : ℚ :=
  (counterfeit_notes.choose 2 : ℚ) / (total_notes.choose 2 : ℚ)

/-- Calculates the probability that at least one drawn note is counterfeit -/
def prob_at_least_one_counterfeit : ℚ :=
  ((counterfeit_notes.choose 2 + counterfeit_notes * genuine_notes) : ℚ) / (total_notes.choose 2 : ℚ)

/-- The main theorem to be proved -/
theorem probability_other_note_counterfeit :
  prob_both_counterfeit / prob_at_least_one_counterfeit = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_other_note_counterfeit_l2266_226625


namespace NUMINAMATH_CALUDE_triangle_inradius_l2266_226647

/-- Given a triangle with perimeter 32 cm and area 40 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : p = 32) 
  (h_area : A = 40) 
  (h_inradius : A = r * p / 2) : 
  r = 2.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_inradius_l2266_226647


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2266_226636

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℝ),
  (x₁ = Real.sqrt 3 ∧ x₂ = (Real.sqrt 3) / 3 ∧ x₃ = -(2 * Real.sqrt 3)) ∧
  (x₁ * x₂ = 1) ∧
  (3 * x₁^3 + 2 * Real.sqrt 3 * x₁^2 - 21 * x₁ + 6 * Real.sqrt 3 = 0) ∧
  (3 * x₂^3 + 2 * Real.sqrt 3 * x₂^2 - 21 * x₂ + 6 * Real.sqrt 3 = 0) ∧
  (3 * x₃^3 + 2 * Real.sqrt 3 * x₃^2 - 21 * x₃ + 6 * Real.sqrt 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2266_226636


namespace NUMINAMATH_CALUDE_solution_to_equation_l2266_226631

theorem solution_to_equation : 
  {x : ℝ | Real.sqrt ((3 + Real.sqrt 8) ^ x) + Real.sqrt ((3 - Real.sqrt 8) ^ x) = 6} = {2, -2} := by
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2266_226631


namespace NUMINAMATH_CALUDE_train_length_l2266_226653

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → 
  time_s = 2.49980001599872 → 
  ∃ (length_m : ℝ), abs (length_m - 99.992) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2266_226653


namespace NUMINAMATH_CALUDE_sean_needs_six_packs_l2266_226651

/-- The number of light bulbs Sean needs to replace in each room --/
def bulbs_per_room : List Nat := [2, 1, 1, 4]

/-- The number of bulbs per pack --/
def bulbs_per_pack : Nat := 2

/-- The fraction of the total bulbs needed for the garage --/
def garage_fraction : Rat := 1/2

/-- Theorem: Sean needs 6 packs of light bulbs --/
theorem sean_needs_six_packs :
  let total_bulbs := (List.sum bulbs_per_room) + ⌈(List.sum bulbs_per_room : Rat) * garage_fraction⌉
  ⌈(total_bulbs : Rat) / bulbs_per_pack⌉ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sean_needs_six_packs_l2266_226651


namespace NUMINAMATH_CALUDE_lune_area_zero_l2266_226660

/-- The area of a lune formed by a semicircle of diameter 2 sitting on top of a semicircle of diameter 4 is 0 -/
theorem lune_area_zero (π : ℝ) (h : π > 0) : 
  let small_semicircle_area := (1/2) * π * (2/2)^2
  let large_semicircle_sector_area := (1/8) * π * (4/2)^2
  let lune_area := small_semicircle_area - large_semicircle_sector_area
  lune_area = 0 := by sorry

end NUMINAMATH_CALUDE_lune_area_zero_l2266_226660


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2266_226654

theorem complex_number_quadrant : ∃ (z : ℂ), z = (Complex.I : ℂ) / (1 + Complex.I) ∧ 0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2266_226654


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_0008_l2266_226646

theorem scientific_notation_of_0_0008 : ∃ (a : ℝ) (n : ℤ), 
  0.0008 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_0008_l2266_226646


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l2266_226616

theorem sum_of_fifth_powers (x y z : ℝ) 
  (eq1 : x + y + z = 3)
  (eq2 : x^3 + y^3 + z^3 = 15)
  (eq3 : x^4 + y^4 + z^4 = 35)
  (ineq : x^2 + y^2 + z^2 < 10) :
  x^5 + y^5 + z^5 = 83 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l2266_226616


namespace NUMINAMATH_CALUDE_unique_solution_l2266_226676

theorem unique_solution : ∃! x : ℝ, 
  -1 < x ∧ x ≤ 2 ∧ 
  Real.sqrt (2 - x) + Real.sqrt (2 + 2*x) = Real.sqrt ((x^4 + 1)/(x^2 + 1)) + (x + 3)/(x + 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2266_226676


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l2266_226608

theorem halfway_between_fractions : 
  let a := (1 : ℚ) / 8
  let b := (1 : ℚ) / 3
  (a + b) / 2 = 11 / 48 := by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l2266_226608


namespace NUMINAMATH_CALUDE_intersection_A_B_l2266_226666

def A : Set ℝ := {x : ℝ | |x| < 3}
def B : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2266_226666


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_four_l2266_226665

theorem reciprocal_of_negative_four :
  ∃ x : ℚ, x * (-4) = 1 ∧ x = -1/4 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_four_l2266_226665


namespace NUMINAMATH_CALUDE_triangles_similar_l2266_226669

/-- A triangle with side lengths a, b, and c. -/
structure Triangle :=
  (a b c : ℝ)
  (positive_a : a > 0)
  (positive_b : b > 0)
  (positive_c : c > 0)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

/-- The condition that a + c = 2b for a triangle. -/
def condition1 (t : Triangle) : Prop :=
  t.a + t.c = 2 * t.b

/-- The condition that b + 2c = 5a for a triangle. -/
def condition2 (t : Triangle) : Prop :=
  t.b + 2 * t.c = 5 * t.a

/-- Two triangles are similar. -/
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.a = k * t2.a ∧ t1.b = k * t2.b ∧ t1.c = k * t2.c

/-- 
Theorem: If two triangles satisfy both condition1 and condition2, then they are similar.
-/
theorem triangles_similar (t1 t2 : Triangle) 
  (h1 : condition1 t1) (h2 : condition1 t2) 
  (h3 : condition2 t1) (h4 : condition2 t2) : 
  similar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_triangles_similar_l2266_226669


namespace NUMINAMATH_CALUDE_sequence_properties_l2266_226694

def sequence_a (n : ℕ) : ℝ := 1 - 2^n

def sum_S (n : ℕ) : ℝ := n + 2 - 2^(n+1)

theorem sequence_properties :
  ∀ (n : ℕ), n ≥ 1 → 
  (∃ (a : ℕ → ℝ) (S : ℕ → ℝ), 
    (∀ k, k ≥ 1 → S k = 2 * a k + k) ∧ 
    (∃ r : ℝ, ∀ k, k ≥ 1 → a (k+1) - 1 = r * (a k - 1)) ∧
    (∀ k, k ≥ 1 → a k = sequence_a k) ∧
    (∀ k, k ≥ 1 → S k = sum_S k)) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2266_226694


namespace NUMINAMATH_CALUDE_even_square_sum_perfect_square_l2266_226686

theorem even_square_sum_perfect_square (x y z : ℤ) 
  (h_even : Even x)
  (h_odd : Odd y)
  (h_sum : x^2 + y^2 = z^2) :
  4 ∣ x :=
sorry

end NUMINAMATH_CALUDE_even_square_sum_perfect_square_l2266_226686


namespace NUMINAMATH_CALUDE_min_center_value_l2266_226680

def RegularOctagon (vertices : Fin 8 → ℕ) (center : ℕ) :=
  (∀ i j : Fin 8, i ≠ j → vertices i ≠ vertices j) ∧
  (vertices 0 + vertices 1 + vertices 4 + vertices 5 + center =
   vertices 1 + vertices 2 + vertices 5 + vertices 6 + center) ∧
  (vertices 2 + vertices 3 + vertices 6 + vertices 7 + center =
   vertices 3 + vertices 0 + vertices 7 + vertices 4 + center) ∧
  (vertices 0 + vertices 1 + vertices 2 + vertices 3 +
   vertices 4 + vertices 5 + vertices 6 + vertices 7 =
   vertices 0 + vertices 1 + vertices 4 + vertices 5 + center)

theorem min_center_value (vertices : Fin 8 → ℕ) (center : ℕ) 
  (h : RegularOctagon vertices center) :
  center ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_min_center_value_l2266_226680


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2266_226607

def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2266_226607


namespace NUMINAMATH_CALUDE_limit_special_function_l2266_226601

/-- The limit of (7^(3x) - 3^(2x)) / (tan(x) + x^3) as x approaches 0 is ln(343/9) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ →
    |((7^(3*x) - 3^(2*x)) / (Real.tan x + x^3)) - Real.log (343/9)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_special_function_l2266_226601


namespace NUMINAMATH_CALUDE_lucy_age_theorem_l2266_226621

/-- Lucy's age at the end of 2000 -/
def lucy_age_2000 : ℝ := 27.5

/-- Lucy's grandfather's age at the end of 2000 -/
def grandfather_age_2000 : ℝ := 3 * lucy_age_2000

/-- The sum of Lucy's and her grandfather's birth years -/
def birth_years_sum : ℝ := 3890

/-- Lucy's age at the end of 2010 -/
def lucy_age_2010 : ℝ := lucy_age_2000 + 10

theorem lucy_age_theorem :
  lucy_age_2000 = (grandfather_age_2000 / 3) ∧
  (2000 - lucy_age_2000) + (2000 - grandfather_age_2000) = birth_years_sum ∧
  lucy_age_2010 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_lucy_age_theorem_l2266_226621


namespace NUMINAMATH_CALUDE_total_ways_to_draw_l2266_226650

/-- Represents the number of cards of each color -/
def cards_per_color : ℕ := 5

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the total number of cards -/
def total_cards : ℕ := cards_per_color * num_colors

/-- Represents the number of cards to be drawn -/
def cards_to_draw : ℕ := 5

/-- Represents the number of ways to draw cards in the (3,1,1) distribution -/
def ways_311 : ℕ := (Nat.choose 3 1) * (Nat.choose cards_per_color 3) * (Nat.choose 2 1) * (Nat.choose 2 1) / 2

/-- Represents the number of ways to draw cards in the (2,2,1) distribution -/
def ways_221 : ℕ := (Nat.choose 3 1) * (Nat.choose cards_per_color 2) * (Nat.choose 2 1) * (Nat.choose 3 2) * (Nat.choose 1 1) / 2

/-- The main theorem stating the total number of ways to draw the cards -/
theorem total_ways_to_draw : ways_311 + ways_221 = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_to_draw_l2266_226650


namespace NUMINAMATH_CALUDE_directrix_of_parabola_l2266_226693

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- State the theorem
theorem directrix_of_parabola :
  ∀ x y : ℝ, parabola x y → (∃ p : ℝ, x = -3 ∧ p = y) :=
by sorry

end NUMINAMATH_CALUDE_directrix_of_parabola_l2266_226693


namespace NUMINAMATH_CALUDE_complex_distance_l2266_226614

theorem complex_distance (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 3)
  (h2 : Complex.abs z₂ = 4)
  (h3 : Complex.abs (z₁ + z₂) = 5) :
  Complex.abs (z₁ - z₂) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_l2266_226614


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2266_226691

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 22*x^2 + 80*x - 67

-- Define the roots
variables (p q r : ℝ)

-- Define A, B, C
variables (A B C : ℝ)

-- Axioms
axiom distinct_roots : p ≠ q ∧ q ≠ r ∧ p ≠ r
axiom roots : f p = 0 ∧ f q = 0 ∧ f r = 0

axiom partial_fraction_decomposition :
  ∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    1 / (s^3 - 22*s^2 + 80*s - 67) = A / (s - p) + B / (s - q) + C / (s - r)

-- Theorem to prove
theorem sum_of_reciprocals : 1/A + 1/B + 1/C = 244 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2266_226691


namespace NUMINAMATH_CALUDE_angle_half_quadrant_l2266_226658

-- Define the angle α
def α : ℝ := sorry

-- Define the integer k
def k : ℤ := sorry

-- Define the condition for α
axiom α_condition : 40 + k * 360 < α ∧ α < 140 + k * 360

-- Define the first quadrant
def first_quadrant (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 90

-- Define the third quadrant
def third_quadrant (θ : ℝ) : Prop := 180 ≤ θ ∧ θ < 270

-- State the theorem
theorem angle_half_quadrant : 
  first_quadrant (α / 2) ∨ third_quadrant (α / 2) := by sorry

end NUMINAMATH_CALUDE_angle_half_quadrant_l2266_226658


namespace NUMINAMATH_CALUDE_min_sin4_plus_2cos4_l2266_226661

theorem min_sin4_plus_2cos4 (x : ℝ) : 
  (Real.sin x)^4 + 2 * (Real.cos x)^4 ≥ (1/2 : ℝ) :=
by
  sorry

#check min_sin4_plus_2cos4

end NUMINAMATH_CALUDE_min_sin4_plus_2cos4_l2266_226661


namespace NUMINAMATH_CALUDE_problem_solution_l2266_226635

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m / x - 3 / (x^2) - 1

theorem problem_solution :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, Real.exp 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ m, (∀ x, 0 < x → 2 * f x ≥ g m x) ↔ m ≤ 4) ∧
  (∀ x, 0 < x → Real.log x < (2 * x / Real.exp 1) - (x^2 / Real.exp x)) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2266_226635


namespace NUMINAMATH_CALUDE_sqrt_neg_one_is_plus_minus_i_l2266_226699

theorem sqrt_neg_one_is_plus_minus_i :
  ∃ (z : ℂ), z * z = -1 ∧ (z = Complex.I ∨ z = -Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_neg_one_is_plus_minus_i_l2266_226699


namespace NUMINAMATH_CALUDE_rectangle_horizontal_length_l2266_226679

/-- Represents the properties of a rectangle --/
structure Rectangle where
  vertical_length : ℝ
  horizontal_length : ℝ
  perimeter : ℝ

/-- Theorem: A rectangle with perimeter 54 cm and horizontal length 3 cm longer than vertical length has a horizontal length of 15 cm --/
theorem rectangle_horizontal_length 
  (rect : Rectangle) 
  (h_perimeter : rect.perimeter = 54)
  (h_length_diff : rect.horizontal_length = rect.vertical_length + 3) : 
  rect.horizontal_length = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_horizontal_length_l2266_226679


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2266_226615

/-- Represents a digit from 1 to 6 -/
def Digit := Fin 6

/-- Represents a two-digit number composed of two digits -/
def TwoDigitNumber (a b : Digit) : ℕ := (a.val + 1) * 10 + (b.val + 1)

/-- The main theorem stating the existence and uniqueness of the solution -/
theorem unique_solution_exists :
  ∃! (A B C D E F : Digit),
    (TwoDigitNumber A B) ^ (C.val + 1) = (TwoDigitNumber D E) ^ (F.val + 1) ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2266_226615


namespace NUMINAMATH_CALUDE_distance_to_big_rock_l2266_226673

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock (v : ℝ) (c : ℝ) (t : ℝ) (h1 : v = 6) (h2 : c = 1) (h3 : t = 1) :
  ∃ d : ℝ, d = 35 / 12 ∧ d / (v - c) + d / (v + c) = t :=
by sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_l2266_226673


namespace NUMINAMATH_CALUDE_states_fraction_1790_1799_l2266_226610

theorem states_fraction_1790_1799 (total_states : ℕ) (states_1790_1799 : ℕ) :
  total_states = 30 →
  states_1790_1799 = 9 →
  (states_1790_1799 : ℚ) / total_states = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_states_fraction_1790_1799_l2266_226610


namespace NUMINAMATH_CALUDE_move_fulcrum_towards_wood_l2266_226628

/-- Represents the material of a sphere -/
inductive Material
| CastIron
| Wood

/-- Properties of a sphere -/
structure Sphere where
  material : Material
  density : ℝ
  volume : ℝ
  mass : ℝ

/-- The setup of the balance problem -/
structure BalanceSetup where
  airDensity : ℝ
  castIronSphere : Sphere
  woodenSphere : Sphere
  fulcrumPosition : ℝ  -- 0 means middle, negative means towards cast iron, positive means towards wood

/-- Conditions for the balance problem -/
def validSetup (setup : BalanceSetup) : Prop :=
  setup.castIronSphere.material = Material.CastIron ∧
  setup.woodenSphere.material = Material.Wood ∧
  setup.castIronSphere.density > setup.woodenSphere.density ∧
  setup.castIronSphere.density > setup.airDensity ∧
  setup.woodenSphere.density > setup.airDensity ∧
  setup.castIronSphere.volume < setup.woodenSphere.volume ∧
  setup.castIronSphere.mass < setup.woodenSphere.mass

/-- The balance condition when the fulcrum is in the middle -/
def balanceCondition (setup : BalanceSetup) : Prop :=
  (setup.castIronSphere.density - setup.airDensity) * setup.castIronSphere.volume =
  (setup.woodenSphere.density - setup.airDensity) * setup.woodenSphere.volume

/-- Theorem stating that the fulcrum needs to be moved towards the wooden sphere -/
theorem move_fulcrum_towards_wood (setup : BalanceSetup) :
  validSetup setup → balanceCondition setup → setup.fulcrumPosition > 0 := by
  sorry

end NUMINAMATH_CALUDE_move_fulcrum_towards_wood_l2266_226628


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l2266_226681

def g (x : ℝ) : ℝ := -3 * x^3 + 50 * x^2 - 4 * x + 10

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) := by
sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l2266_226681


namespace NUMINAMATH_CALUDE_first_day_over_threshold_l2266_226626

/-- The number of paperclips Max starts with on Monday -/
def initial_paperclips : ℕ := 3

/-- The factor by which the number of paperclips increases each day -/
def daily_increase_factor : ℕ := 4

/-- The threshold number of paperclips -/
def threshold : ℕ := 200

/-- The function that calculates the number of paperclips on day n -/
def paperclips (n : ℕ) : ℕ := initial_paperclips * daily_increase_factor^(n - 1)

/-- The theorem stating that the 5th day is the first day with more than 200 paperclips -/
theorem first_day_over_threshold :
  ∀ n : ℕ, n > 0 → (paperclips n > threshold ↔ n ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_first_day_over_threshold_l2266_226626


namespace NUMINAMATH_CALUDE_gcd_of_36_45_495_l2266_226698

theorem gcd_of_36_45_495 : Nat.gcd 36 (Nat.gcd 45 495) = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_36_45_495_l2266_226698


namespace NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l2266_226627

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 3}
def B : Set Nat := {3, 5}

theorem intersection_complement_equals_singleton : A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l2266_226627


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l2266_226655

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l2266_226655


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2266_226641

theorem sum_of_numbers : 1357 + 3571 + 5713 + 7135 + 1357 = 19133 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2266_226641


namespace NUMINAMATH_CALUDE_x_plus_y_equals_three_l2266_226670

theorem x_plus_y_equals_three (x y : ℝ) (h : |x - 1| + (y - 2)^2 = 0) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_three_l2266_226670


namespace NUMINAMATH_CALUDE_committee_count_l2266_226690

theorem committee_count (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 1) :
  (Nat.choose (n - k) (m - k)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l2266_226690


namespace NUMINAMATH_CALUDE_complement_of_A_l2266_226675

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 ≥ 0}

-- State the theorem
theorem complement_of_A :
  (Set.univ : Set ℝ) \ A = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2266_226675


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2266_226602

theorem cubic_roots_sum_cubes (u v w : ℝ) : 
  (5 * u^3 + 500 * u + 1005 = 0) →
  (5 * v^3 + 500 * v + 1005 = 0) →
  (5 * w^3 + 500 * w + 1005 = 0) →
  (u + v)^3 + (v + w)^3 + (w + u)^3 = 603 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2266_226602


namespace NUMINAMATH_CALUDE_student_lecture_selections_l2266_226659

/-- The number of different selection methods for students choosing lectures -/
def selection_methods (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: Given 4 students and 3 lectures, the number of different selection methods is 81 -/
theorem student_lecture_selections :
  selection_methods 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_student_lecture_selections_l2266_226659


namespace NUMINAMATH_CALUDE_multiply_48_52_l2266_226638

theorem multiply_48_52 : 48 * 52 = 2496 := by
  sorry

end NUMINAMATH_CALUDE_multiply_48_52_l2266_226638


namespace NUMINAMATH_CALUDE_quadrilateral_on_exponential_curve_l2266_226623

theorem quadrilateral_on_exponential_curve (e : ℝ) (h_e : e > 0) :
  ∃ m : ℕ+, 
    (1/2 * (e^(m : ℝ) - e^((m : ℝ) + 3)) = (e^2 - 1) / e) ∧ 
    (∀ k : ℕ+, k < m → 1/2 * (e^(k : ℝ) - e^((k : ℝ) + 3)) ≠ (e^2 - 1) / e) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_on_exponential_curve_l2266_226623


namespace NUMINAMATH_CALUDE_percentage_of_population_l2266_226682

theorem percentage_of_population (total_population : ℕ) (part_population : ℕ) :
  total_population = 28800 →
  part_population = 23040 →
  (part_population : ℚ) / (total_population : ℚ) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_population_l2266_226682


namespace NUMINAMATH_CALUDE_number_solution_l2266_226632

theorem number_solution : ∃ x : ℝ, 45 - 3 * x = 12 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l2266_226632


namespace NUMINAMATH_CALUDE_triangle_acute_from_angle_ratio_l2266_226677

/-- Theorem: In a triangle ABC where the ratio of angles A:B:C is 2:3:4, all angles are less than 90 degrees. -/
theorem triangle_acute_from_angle_ratio (A B C : ℝ) (h_ratio : ∃ (x : ℝ), A = 2*x ∧ B = 3*x ∧ C = 4*x) 
  (h_sum : A + B + C = 180) : A < 90 ∧ B < 90 ∧ C < 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_acute_from_angle_ratio_l2266_226677


namespace NUMINAMATH_CALUDE_general_solution_zero_a_case_degenerate_case_l2266_226634

-- Define the system of equations
def system (a b c x y z : ℝ) : Prop :=
  a * x + b * y - c * z = a * b ∧
  3 * a * x - b * y + 2 * c * z = a * (5 * c - b) ∧
  3 * y + 2 * z = 5 * a

-- Theorem for the general solution
theorem general_solution (a b c : ℝ) :
  ∃ x y z, system a b c x y z ∧ x = c ∧ y = a ∧ z = a :=
sorry

-- Theorem for the case when a = 0
theorem zero_a_case (b c : ℝ) :
  ∃ x y z, system 0 b c x y z ∧ y = 0 ∧ z = 0 :=
sorry

-- Theorem for the case when 8b + 15c = 0
theorem degenerate_case (a b : ℝ) :
  8 * b + 15 * (-8 * b / 15) = 0 →
  ∃ x y, ∀ z, system a b (-8 * b / 15) x y z :=
sorry

end NUMINAMATH_CALUDE_general_solution_zero_a_case_degenerate_case_l2266_226634


namespace NUMINAMATH_CALUDE_smallest_palindromic_prime_l2266_226643

/-- A function that checks if a number is a three-digit palindrome with hundreds digit 2 -/
def isValidPalindrome (n : ℕ) : Prop :=
  n ≥ 200 ∧ n ≤ 299 ∧ (n / 100 = 2) ∧ (n % 10 = n / 100)

/-- The theorem stating that 232 is the smallest three-digit palindromic prime with hundreds digit 2 -/
theorem smallest_palindromic_prime :
  isValidPalindrome 232 ∧ Nat.Prime 232 ∧
  ∀ n < 232, isValidPalindrome n → ¬Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_smallest_palindromic_prime_l2266_226643


namespace NUMINAMATH_CALUDE_mode_of_throws_l2266_226657

def throw_results : List Float := [7.6, 8.5, 8.6, 8.5, 9.1, 8.5, 8.4, 8.6, 9.2, 7.3]

def mode (l : List Float) : Float :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_throws :
  mode throw_results = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_throws_l2266_226657


namespace NUMINAMATH_CALUDE_triangle_theorem_l2266_226603

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - Real.sqrt 2 * t.b * t.c = t.a^2)
  (h2 : t.c / t.b = 2 * Real.sqrt 2) : 
  t.A = π/4 ∧ Real.tan t.B = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2266_226603


namespace NUMINAMATH_CALUDE_min_editors_conference_l2266_226644

theorem min_editors_conference (total : ℕ) (writers : ℕ) (max_both : ℕ) :
  total = 100 →
  writers = 35 →
  max_both = 26 →
  ∃ (editors : ℕ) (both : ℕ),
    both ≤ max_both ∧
    editors ≥ 39 ∧
    total = writers + editors - both + 2 * both :=
by
  sorry

end NUMINAMATH_CALUDE_min_editors_conference_l2266_226644


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2266_226671

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ -1 ∧ x^2 - x + 2 ≠ 0 →
  (x^2 + 2*x - 8) / (x^3 - x - 2) = 
  (-9/4) / (x + 1) + (13/4 * x - 7/2) / (x^2 - x + 2) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2266_226671


namespace NUMINAMATH_CALUDE_beads_per_necklace_l2266_226683

theorem beads_per_necklace (total_beads : ℕ) (num_necklaces : ℕ) 
  (h1 : total_beads = 28) (h2 : num_necklaces = 4) :
  total_beads / num_necklaces = 7 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l2266_226683


namespace NUMINAMATH_CALUDE_unique_sum_with_identical_digits_l2266_226624

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Checks if a number is a three-digit number with identical digits -/
def is_three_identical_digits (m : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ Finset.range 10 ∧ m = 111 * d

theorem unique_sum_with_identical_digits :
  ∃! (n : ℕ), is_three_identical_digits (sum_of_first_n n) :=
sorry

end NUMINAMATH_CALUDE_unique_sum_with_identical_digits_l2266_226624


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2266_226662

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x - 2 < 0) ↔ a ∈ Set.Ioc (-2) 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2266_226662


namespace NUMINAMATH_CALUDE_functional_inequality_solution_l2266_226697

open Real

-- Define the function type
def ContinuousRealFunction := {f : ℝ → ℝ // Continuous f}

-- State the theorem
theorem functional_inequality_solution 
  (f : ContinuousRealFunction) 
  (h1 : f.val 0 = 0) 
  (h2 : ∀ x y : ℝ, f.val ((x + y) / (1 + x * y)) ≥ f.val x + f.val y) :
  ∃ c : ℝ, ∀ x : ℝ, f.val x = (c / 2) * log (abs ((x + 1) / (x - 1))) :=
sorry

end NUMINAMATH_CALUDE_functional_inequality_solution_l2266_226697


namespace NUMINAMATH_CALUDE_geometric_mean_sqrt2_plus_minus_one_l2266_226620

theorem geometric_mean_sqrt2_plus_minus_one : 
  Real.sqrt ((Real.sqrt 2 + 1) * (Real.sqrt 2 - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_sqrt2_plus_minus_one_l2266_226620


namespace NUMINAMATH_CALUDE_line_arrangements_l2266_226617

/-- The number of different arrangements for 3 boys and 4 girls standing in a line under various conditions -/
theorem line_arrangements (n : ℕ) (boys : ℕ) (girls : ℕ) : 
  boys = 3 → girls = 4 → n = boys + girls →
  (∃ (arrangements_1 arrangements_2 arrangements_3 arrangements_4 : ℕ),
    /- Condition 1: Person A and B must stand at the two ends -/
    arrangements_1 = 240 ∧
    /- Condition 2: Person A cannot stand at the left end, and person B cannot stand at the right end -/
    arrangements_2 = 3720 ∧
    /- Condition 3: Person A and B must stand next to each other -/
    arrangements_3 = 1440 ∧
    /- Condition 4: The 3 boys are arranged from left to right in descending order of height -/
    arrangements_4 = 840) :=
by sorry

end NUMINAMATH_CALUDE_line_arrangements_l2266_226617


namespace NUMINAMATH_CALUDE_distance_calculation_l2266_226687

/-- Proves that the distance run by A and B is 2250 meters given their running times and the difference in distance covered. -/
theorem distance_calculation (D : ℝ) 
  (h1 : D / 90 * 180 = D + 2250) : D = 2250 := by
  sorry

#check distance_calculation

end NUMINAMATH_CALUDE_distance_calculation_l2266_226687


namespace NUMINAMATH_CALUDE_some_students_not_club_members_l2266_226674

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (ClubMember : U → Prop)
variable (Dishonest : U → Prop)

-- Define the conditions
variable (some_students_dishonest : ∃ x, Student x ∧ Dishonest x)
variable (all_club_members_honest : ∀ x, ClubMember x → ¬Dishonest x)

-- Theorem to prove
theorem some_students_not_club_members :
  ∃ x, Student x ∧ ¬ClubMember x :=
sorry

end NUMINAMATH_CALUDE_some_students_not_club_members_l2266_226674


namespace NUMINAMATH_CALUDE_modulus_of_z_l2266_226663

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 3 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2266_226663


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2266_226649

theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ),
    ∀ (x : ℝ), x ≠ 4 ∧ x ≠ 3 ∧ x ≠ 5 →
      (x^2 - 5) / ((x - 4) * (x - 3) * (x - 5)) =
      A / (x - 4) + B / (x - 3) + C / (x - 5) ↔
      A = -11 ∧ B = 2 ∧ C = 10 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2266_226649


namespace NUMINAMATH_CALUDE_leadership_structure_count_15_l2266_226656

/-- The number of ways to select a leadership structure from a group of people. -/
def leadershipStructureCount (n : ℕ) : ℕ :=
  n * (n - 1).choose 2 * (n - 3).choose 3 * (n - 6).choose 3

/-- Theorem stating that the number of ways to select a leadership structure
    from 15 people is 2,717,880. -/
theorem leadership_structure_count_15 :
  leadershipStructureCount 15 = 2717880 := by
  sorry

end NUMINAMATH_CALUDE_leadership_structure_count_15_l2266_226656


namespace NUMINAMATH_CALUDE_system_solution_existence_l2266_226600

theorem system_solution_existence (m : ℝ) : 
  (m ≠ 1) ↔ (∃ (x y : ℝ), y = m * x + 5 ∧ y = (3 * m - 2) * x + 7) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l2266_226600


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l2266_226619

/-- Given a quadratic function f(x) = x^2 + 1800x + 1800, 
    we can rewrite it in the form (x+b)^2 + c.
    This theorem states that the ratio c/b equals -898. -/
theorem quadratic_rewrite_ratio : 
  ∃ (b c : ℝ), (∀ x, x^2 + 1800*x + 1800 = (x+b)^2 + c) ∧ (c/b = -898) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l2266_226619


namespace NUMINAMATH_CALUDE_gcd_48576_34650_l2266_226645

theorem gcd_48576_34650 : Nat.gcd 48576 34650 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_48576_34650_l2266_226645


namespace NUMINAMATH_CALUDE_games_played_together_count_l2266_226611

/-- The number of players in the league -/
def totalPlayers : ℕ := 12

/-- The number of players in each game -/
def playersPerGame : ℕ := 6

/-- Function to calculate the number of games two specific players play together -/
def gamesPlayedTogether : ℕ := sorry

theorem games_played_together_count :
  gamesPlayedTogether = 210 := by sorry

end NUMINAMATH_CALUDE_games_played_together_count_l2266_226611


namespace NUMINAMATH_CALUDE_bob_rock_skips_bob_rock_skips_solution_l2266_226612

theorem bob_rock_skips (jim_skips : ℕ) (rocks_each : ℕ) (total_skips : ℕ) : ℕ :=
  let bob_skips := (total_skips - jim_skips * rocks_each) / rocks_each
  bob_skips

#check @bob_rock_skips

theorem bob_rock_skips_solution :
  bob_rock_skips 15 10 270 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bob_rock_skips_bob_rock_skips_solution_l2266_226612


namespace NUMINAMATH_CALUDE_equal_roots_condition_l2266_226613

/-- The quadratic equation x^2 - nx + 9 = 0 has two equal real roots if and only if n = 6 or n = -6 -/
theorem equal_roots_condition (n : ℝ) : 
  (∃ x : ℝ, x^2 - n*x + 9 = 0 ∧ (∀ y : ℝ, y^2 - n*y + 9 = 0 → y = x)) ↔ 
  (n = 6 ∨ n = -6) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l2266_226613


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l2266_226695

/-- A function satisfying the given inequality -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + y ≤ f (f (f x))

/-- The main theorem stating the form of functions satisfying the inequality -/
theorem characterize_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfiesInequality f →
  ∃ C : ℝ, ∀ x : ℝ, f x = -x + C :=
by sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l2266_226695


namespace NUMINAMATH_CALUDE_constant_value_l2266_226642

theorem constant_value (t : ℝ) (x y : ℝ → ℝ) (c : ℝ) :
  (∀ t, x t = c - 4 * t) →
  (∀ t, y t = 2 * t - 2) →
  x 0.5 = y 0.5 →
  c = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l2266_226642


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2266_226640

/-- Given a line with equation x - 2y + 1 = 0, prove that the sum of its x-intercept and y-intercept is -1/2 -/
theorem line_intercepts_sum (x y : ℝ) : 
  x - 2*y + 1 = 0 → 
  ∃ (x_int y_int : ℝ), x_int - 2*0 + 1 = 0 ∧ 0 - 2*y_int + 1 = 0 ∧ x_int + y_int = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2266_226640


namespace NUMINAMATH_CALUDE_eleven_divides_reversible_integer_l2266_226689

/-- A 5-digit positive integer with the first three digits the same as its first three digits in reverse order -/
def ReversibleInteger (z : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    0 < a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    z = 10000 * a + 1000 * b + 100 * c + 10 * a + b

theorem eleven_divides_reversible_integer (z : ℕ) (h : ReversibleInteger z) : 
  11 ∣ z :=
sorry

end NUMINAMATH_CALUDE_eleven_divides_reversible_integer_l2266_226689


namespace NUMINAMATH_CALUDE_min_value_polynomial_min_value_achieved_l2266_226629

theorem min_value_polynomial (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 ≥ 2161.75 :=
sorry

theorem min_value_achieved : 
  ∃ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 = 2161.75 :=
sorry

end NUMINAMATH_CALUDE_min_value_polynomial_min_value_achieved_l2266_226629


namespace NUMINAMATH_CALUDE_levels_ratio_l2266_226685

def total_levels : ℕ := 32
def beaten_levels : ℕ := 24

theorem levels_ratio :
  let not_beaten := total_levels - beaten_levels
  (beaten_levels : ℚ) / not_beaten = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_levels_ratio_l2266_226685


namespace NUMINAMATH_CALUDE_simplify_fraction_l2266_226630

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  (1 - 1 / (x - 1)) / ((x^2 - 2*x) / (x^2 - 1)) = (x + 1) / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2266_226630


namespace NUMINAMATH_CALUDE_sum_mod_five_l2266_226605

theorem sum_mod_five : (9375 + 9376 + 9377 + 9378) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_five_l2266_226605


namespace NUMINAMATH_CALUDE_perimeter_of_seven_unit_squares_l2266_226652

/-- A figure composed of unit squares meeting at vertices -/
structure SquareFigure where
  num_squares : ℕ
  squares_meet_at_vertices : Bool

/-- The perimeter of a square figure -/
def perimeter (f : SquareFigure) : ℕ := 
  if f.squares_meet_at_vertices then
    4 * f.num_squares
  else
    sorry  -- We don't handle this case in this problem

theorem perimeter_of_seven_unit_squares : 
  ∀ (f : SquareFigure), f.num_squares = 7 → f.squares_meet_at_vertices → perimeter f = 28 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_seven_unit_squares_l2266_226652


namespace NUMINAMATH_CALUDE_ferry_speed_difference_l2266_226688

/-- Proves that the difference in speed between ferry Q and ferry P is 3 km/h -/
theorem ferry_speed_difference : 
  ∀ (Vp Vq : ℝ) (time_p time_q : ℝ) (distance_p distance_q : ℝ),
  Vp = 6 →  -- Ferry P's speed
  time_p = 3 →  -- Ferry P's travel time
  distance_p = Vp * time_p →  -- Ferry P's distance
  distance_q = 2 * distance_p →  -- Ferry Q's distance is twice Ferry P's
  time_q = time_p + 1 →  -- Ferry Q's travel time is 1 hour longer
  Vq = distance_q / time_q →  -- Ferry Q's speed
  Vq - Vp = 3 := by
sorry

end NUMINAMATH_CALUDE_ferry_speed_difference_l2266_226688


namespace NUMINAMATH_CALUDE_max_triangles_six_lines_l2266_226648

/-- A configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  on_plane : Bool

/-- Counts the number of equilateral triangles formed by line intersections -/
def count_equilateral_triangles (config : LineConfiguration) : ℕ :=
  sorry

/-- The maximum number of equilateral triangles for a given configuration -/
def max_equilateral_triangles (config : LineConfiguration) : ℕ :=
  sorry

/-- Theorem: The maximum number of equilateral triangles formed by six lines on a plane is 8 -/
theorem max_triangles_six_lines :
  ∀ (config : LineConfiguration),
    config.num_lines = 6 ∧ config.on_plane →
    max_equilateral_triangles config = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_triangles_six_lines_l2266_226648


namespace NUMINAMATH_CALUDE_remainder_equivalence_l2266_226604

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem: The remainder when dividing a number by 3 or 9 is the same as
    the remainder when dividing the sum of its digits by 3 or 9 -/
theorem remainder_equivalence (n : ℕ) :
  (n % 3 = sum_of_digits n % 3) ∧ (n % 9 = sum_of_digits n % 9) :=
sorry

end NUMINAMATH_CALUDE_remainder_equivalence_l2266_226604


namespace NUMINAMATH_CALUDE_toy_cost_price_l2266_226633

/-- Given the sale of toys, prove the cost price of a single toy. -/
theorem toy_cost_price (num_sold : ℕ) (total_price : ℕ) (gain_equiv : ℕ) (cost_price : ℕ) :
  num_sold = 36 →
  total_price = 45000 →
  gain_equiv = 6 →
  total_price = num_sold * cost_price + gain_equiv * cost_price →
  cost_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_price_l2266_226633


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2266_226639

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 + x / 3) ^ (1/3 : ℝ) = -4 :=
by
  use -207
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2266_226639


namespace NUMINAMATH_CALUDE_smallest_non_representable_as_cube_sum_l2266_226606

theorem smallest_non_representable_as_cube_sum : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), m < n → ∃ (x y : ℤ), m = x^3 + 3*y^3) ∧
  ¬∃ (x y : ℤ), n = x^3 + 3*y^3 ∧ 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_representable_as_cube_sum_l2266_226606


namespace NUMINAMATH_CALUDE_count_multiples_of_7_ending_in_7_less_than_150_l2266_226668

def multiples_of_7_ending_in_7 (n : ℕ) : ℕ :=
  (n / 70 : ℕ)

theorem count_multiples_of_7_ending_in_7_less_than_150 :
  multiples_of_7_ending_in_7 150 = 2 := by sorry

end NUMINAMATH_CALUDE_count_multiples_of_7_ending_in_7_less_than_150_l2266_226668


namespace NUMINAMATH_CALUDE_inheritance_satisfies_tax_equation_l2266_226672

/-- Represents the inheritance amount in dollars -/
def inheritance : ℝ := sorry

/-- The total tax paid is $15000 -/
def total_tax : ℝ := 15000

/-- Federal tax rate is 25% -/
def federal_tax_rate : ℝ := 0.25

/-- State tax rate is 15% -/
def state_tax_rate : ℝ := 0.15

/-- Theorem stating that the inheritance satisfies the tax equation -/
theorem inheritance_satisfies_tax_equation : 
  federal_tax_rate * inheritance + state_tax_rate * (1 - federal_tax_rate) * inheritance = total_tax := by
  sorry

end NUMINAMATH_CALUDE_inheritance_satisfies_tax_equation_l2266_226672


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2266_226622

/-- Two lines are parallel if their coefficients are proportional -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- The first line l₁: (m+3)x + 4y + 3m - 5 = 0 -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y + 3 * m - 5 = 0

/-- The second line l₂: 2x + (m+5)y - 8 = 0 -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y - 8 = 0

/-- Theorem: If l₁ and l₂ are parallel, then m = -7 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel (m + 3) 4 2 (m + 5) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2266_226622


namespace NUMINAMATH_CALUDE_derivative_of_sqrt_at_one_l2266_226667

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem derivative_of_sqrt_at_one :
  deriv f 1 = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_sqrt_at_one_l2266_226667


namespace NUMINAMATH_CALUDE_inequality_reversal_l2266_226609

theorem inequality_reversal (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l2266_226609


namespace NUMINAMATH_CALUDE_optimal_furniture_purchase_l2266_226692

def maximize_furniture (budget chair_price table_price : ℕ) : ℕ × ℕ :=
  let (tables, chairs) := (25, 37)
  have budget_constraint : tables * table_price + chairs * chair_price ≤ budget := by sorry
  have chair_lower_bound : chairs ≥ tables := by sorry
  have chair_upper_bound : chairs ≤ (3 * tables) / 2 := by sorry
  have is_optimal : ∀ (t c : ℕ), t * table_price + c * chair_price ≤ budget → 
                    c ≥ t → c ≤ (3 * t) / 2 → t + c ≤ tables + chairs := by sorry
  (tables, chairs)

theorem optimal_furniture_purchase :
  let (tables, chairs) := maximize_furniture 2000 20 50
  tables = 25 ∧ chairs = 37 := by sorry

end NUMINAMATH_CALUDE_optimal_furniture_purchase_l2266_226692


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_two_three_l2266_226637

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}

-- Theorem to prove
theorem A_intersect_B_equals_two_three : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_two_three_l2266_226637


namespace NUMINAMATH_CALUDE_min_value_of_expression_limit_at_one_l2266_226664

open Real

theorem min_value_of_expression (x : ℝ) (h1 : -3 < x) (h2 : x < 2) (h3 : x ≠ 1) :
  (x^2 - 4*x + 5) / (3*x - 3) ≥ 2/3 :=
sorry

theorem limit_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |(x^2 - 4*x + 5) / (3*x - 3) - 2/3| < ε :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_limit_at_one_l2266_226664


namespace NUMINAMATH_CALUDE_B_subset_A_iff_m_in_range_l2266_226684

-- Define set A
def A : Set ℝ := {x | (2 * x) / (x - 2) < 1}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m + 1)*x + m^2 + m < 0}

-- Theorem statement
theorem B_subset_A_iff_m_in_range :
  ∀ m : ℝ, (B m) ⊆ A ↔ -2 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_B_subset_A_iff_m_in_range_l2266_226684


namespace NUMINAMATH_CALUDE_sara_score_is_26_l2266_226678

/-- Represents a mathematics contest --/
structure MathContest where
  totalQuestions : Nat
  correctAnswers : Nat
  incorrectAnswers : Nat
  unansweredQuestions : Nat
  pointsPerCorrect : Int
  pointsPerIncorrect : Int
  pointsPerUnanswered : Int

/-- Calculates the total score for a given contest --/
def calculateScore (contest : MathContest) : Int :=
  contest.correctAnswers * contest.pointsPerCorrect +
  contest.incorrectAnswers * contest.pointsPerIncorrect +
  contest.unansweredQuestions * contest.pointsPerUnanswered

/-- Sara's contest performance --/
def sarasContest : MathContest :=
  { totalQuestions := 30
    correctAnswers := 18
    incorrectAnswers := 10
    unansweredQuestions := 2
    pointsPerCorrect := 2
    pointsPerIncorrect := -1
    pointsPerUnanswered := 0
  }

/-- Theorem stating that Sara's score is 26 --/
theorem sara_score_is_26 : calculateScore sarasContest = 26 := by
  sorry

end NUMINAMATH_CALUDE_sara_score_is_26_l2266_226678


namespace NUMINAMATH_CALUDE_problem_statement_l2266_226618

-- Define the logarithm function
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define symmetry about a point
def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

-- Define symmetry about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem problem_statement :
  (∀ a : ℝ, a > 0 → a ≠ 1 → log_base a (a * (-1) + 2 * a) = 1) ∧
  (∃ f : ℝ → ℝ, symmetric_about_origin (fun x ↦ f (x - 3)) ∧
    ¬ symmetric_about f (3, 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2266_226618


namespace NUMINAMATH_CALUDE_evolute_of_ellipse_l2266_226696

/-- The equation of the evolute of an ellipse -/
theorem evolute_of_ellipse (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^2 / a^2 + y^2 / b^2 = 1 →
  (a * x)^(2/3) + (b * y)^(2/3) = (a^2 - b^2)^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_evolute_of_ellipse_l2266_226696
