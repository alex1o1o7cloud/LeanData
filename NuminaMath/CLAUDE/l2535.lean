import Mathlib

namespace reciprocal_of_negative_three_fourths_l2535_253502

theorem reciprocal_of_negative_three_fourths :
  let x : ℚ := -3/4
  let y : ℚ := -4/3
  (x * y = 1) → (y = x⁻¹) := by sorry

end reciprocal_of_negative_three_fourths_l2535_253502


namespace group_size_problem_l2535_253521

theorem group_size_problem (total_collection : ℕ) 
  (h1 : total_collection = 2916) : ∃ n : ℕ, n * n = total_collection ∧ n = 54 := by
  sorry

end group_size_problem_l2535_253521


namespace function_composition_l2535_253568

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem function_composition :
  ∀ x : ℝ, f (g x) = 6 * x - 7 := by
  sorry

end function_composition_l2535_253568


namespace amelia_wins_probability_l2535_253503

/-- Probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 1/4

/-- Probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 1/6

/-- Probability of Amelia winning -/
noncomputable def p_amelia_wins : ℚ := 2/3

/-- Theorem stating that the probability of Amelia winning is 2/3 -/
theorem amelia_wins_probability :
  p_amelia_wins = p_amelia * (1 - p_blaine) + p_amelia * p_blaine + 
  (1 - p_amelia) * (1 - p_blaine) * p_amelia_wins :=
by sorry

end amelia_wins_probability_l2535_253503


namespace absolute_value_inequality_solution_set_l2535_253575

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 1 < |1 - x| ∧ |1 - x| ≤ 2} = Set.Icc (-1) 0 ∪ Set.Ioc 2 3 := by
  sorry

end absolute_value_inequality_solution_set_l2535_253575


namespace tan_function_property_l2535_253589

open Real

theorem tan_function_property (φ a : ℝ) (h1 : π / 2 < φ) (h2 : φ < 3 * π / 2) : 
  let f := fun x => tan (φ - x)
  (f 0 = 0) → (f (-a) = 1 / 2) → (f (a + π / 4) = -3) := by
  sorry

end tan_function_property_l2535_253589


namespace ice_cream_difference_l2535_253515

-- Define the number of scoops for Oli and Victoria
def oli_scoops : ℕ := 4
def victoria_scoops : ℕ := 2 * oli_scoops

-- Theorem statement
theorem ice_cream_difference : victoria_scoops - oli_scoops = 4 := by
  sorry

end ice_cream_difference_l2535_253515


namespace women_fraction_in_room_l2535_253597

theorem women_fraction_in_room (total_people : ℕ) (married_fraction : ℚ) 
  (max_unmarried_women : ℕ) (h1 : total_people = 80) (h2 : married_fraction = 1/2) 
  (h3 : max_unmarried_women = 32) : 
  (max_unmarried_women + (married_fraction * total_people / 2)) / total_people = 1/2 :=
sorry

end women_fraction_in_room_l2535_253597


namespace spending_ratio_l2535_253537

/-- Represents the spending of Lisa and Carly -/
structure Spending where
  lisa_tshirt : ℝ
  lisa_jeans : ℝ
  lisa_coat : ℝ
  carly_tshirt : ℝ
  carly_jeans : ℝ
  carly_coat : ℝ

/-- The theorem representing the problem -/
theorem spending_ratio (s : Spending) : 
  s.lisa_tshirt = 40 →
  s.lisa_jeans = s.lisa_tshirt / 2 →
  s.carly_tshirt = s.lisa_tshirt / 4 →
  s.carly_jeans = 3 * s.lisa_jeans →
  s.carly_coat = s.lisa_coat / 4 →
  s.lisa_tshirt + s.lisa_jeans + s.lisa_coat + 
  s.carly_tshirt + s.carly_jeans + s.carly_coat = 230 →
  s.lisa_coat / s.lisa_tshirt = 2 := by
  sorry

end spending_ratio_l2535_253537


namespace nods_per_kilometer_l2535_253525

/-- Given the relationships between winks, nods, leaps, and kilometers,
    prove that the number of nods in one kilometer is equal to qts / (pru) -/
theorem nods_per_kilometer
  (p q r s t u : ℚ)
  (h1 : p * 1 = q)  -- p winks equal q nods
  (h2 : r * 1 = s)  -- r leaps equal s winks
  (h3 : t * 1 = u)  -- t leaps are equivalent to u kilometers
  : 1 = q * t * s / (p * r * u) :=
sorry

end nods_per_kilometer_l2535_253525


namespace diophantine_equation_implication_l2535_253500

theorem diophantine_equation_implication 
  (a b : ℤ) 
  (ha : ¬∃ (n : ℤ), a = n^2) 
  (hb : ¬∃ (n : ℤ), b = n^2) 
  (h : ∃ (x y z w : ℤ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ w ≠ 0 ∧ x^2 - a*y^2 - b*z^2 + a*b*w^2 = 0) :
  ∃ (X Y Z : ℤ), X ≠ 0 ∨ Y ≠ 0 ∨ Z ≠ 0 ∧ X^2 - a*Y^2 - b*Z^2 = 0 :=
sorry

end diophantine_equation_implication_l2535_253500


namespace average_of_two_numbers_l2535_253546

theorem average_of_two_numbers (a b c : ℝ) : 
  (a + b + c) / 3 = 48 → c = 32 → (a + b) / 2 = 56 := by
sorry

end average_of_two_numbers_l2535_253546


namespace rectangle_cutting_l2535_253583

theorem rectangle_cutting (large_width large_height small_width small_height : ℝ) 
  (hw : large_width = 50)
  (hh : large_height = 90)
  (hsw : small_width = 1)
  (hsh : small_height = 10 * Real.sqrt 2) :
  ⌊(large_width * large_height) / (small_width * small_height)⌋ = 318 := by
  sorry

end rectangle_cutting_l2535_253583


namespace main_theorem_l2535_253524

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The sequence a_n -/
noncomputable def a : Sequence := sorry

/-- The sequence b_n -/
noncomputable def b (n : ℕ) : ℝ := a n + n

/-- The sum of the first n terms of b_n -/
noncomputable def S (n : ℕ) : ℝ := sorry

/-- Main theorem -/
theorem main_theorem :
  (∀ n : ℕ, a n < 0) ∧
  (∀ n : ℕ, a (n + 1) = 2/3 * a n) ∧
  (a 2 * a 5 = 8/27) →
  (∀ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2/3) ∧
  (∀ n : ℕ, a n = -(2/3)^(n-1)) ∧
  (∀ n : ℕ, S n = (n^2 + n + 6)/2 - 3 * (2/3)^n) :=
by sorry

end main_theorem_l2535_253524


namespace selection_theorem_l2535_253509

/-- The number of candidates --/
def n : ℕ := 5

/-- The number of languages --/
def k : ℕ := 3

/-- The number of candidates unwilling to study Hebrew --/
def m : ℕ := 2

/-- The number of ways to select students for the languages --/
def selection_methods : ℕ := (n - m) * (Nat.choose (n - 1) (k - 1)) * 2

theorem selection_theorem : selection_methods = 36 := by
  sorry

end selection_theorem_l2535_253509


namespace parabola_point_coordinates_l2535_253590

/-- A point on a parabola with a specific distance to the focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y ^ 2 = 12 * x
  distance_to_focus : 6 = |x + 3| -- The focus is at (3, 0)

/-- The coordinates of a point on the parabola y² = 12x with distance 6 to the focus -/
theorem parabola_point_coordinates (p : ParabolaPoint) : 
  (p.x = 3 ∧ p.y = 6) ∨ (p.x = 3 ∧ p.y = -6) := by
  sorry

end parabola_point_coordinates_l2535_253590


namespace fraction_sum_equality_l2535_253549

theorem fraction_sum_equality : 
  (3 : ℚ) / 5 + (2 : ℚ) / 3 + (1 + (1 : ℚ) / 15) = 2 + (1 : ℚ) / 3 := by
  sorry

end fraction_sum_equality_l2535_253549


namespace parabola_perpendicular_chords_locus_l2535_253541

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a chord of the parabola -/
structure Chord where
  slope : ℝ

/-- The locus of point M -/
def locus (p : ℝ) (x y : ℝ) : Prop :=
  (x - 2*p)^2 + y^2 = 4*p^2

theorem parabola_perpendicular_chords_locus 
  (para : Parabola) 
  (chord1 chord2 : Chord) 
  (O M : Point) :
  O.x = 0 ∧ O.y = 0 ∧  -- Vertex O at origin
  (chord1.slope * chord2.slope = -1) →  -- Perpendicular chords
  locus para.p M.x M.y  -- Locus of projection M
  := by sorry

end parabola_perpendicular_chords_locus_l2535_253541


namespace bisection_method_termination_condition_l2535_253578

/-- The bisection method termination condition -/
def bisection_termination (x₁ x₂ ε : ℝ) : Prop :=
  |x₁ - x₂| < ε

/-- Theorem stating the correct termination condition for the bisection method -/
theorem bisection_method_termination_condition 
  (f : ℝ → ℝ) (a b x₁ x₂ ε : ℝ) 
  (hf : Continuous f) 
  (ha : f a < 0) 
  (hb : f b > 0) 
  (hε : ε > 0) 
  (hx₁ : x₁ ∈ Set.Icc a b) 
  (hx₂ : x₂ ∈ Set.Icc a b) :
  bisection_termination x₁ x₂ ε ↔ 
    (∃ x ∈ Set.Icc x₁ x₂, f x = 0) ∧ 
    (∀ y ∈ Set.Icc a b, f y = 0 → y ∈ Set.Icc x₁ x₂) := by
  sorry


end bisection_method_termination_condition_l2535_253578


namespace calculation_proof_l2535_253567

theorem calculation_proof : 3.6 * 0.25 + 1.5 = 2.4 := by
  sorry

end calculation_proof_l2535_253567


namespace problem_solution_l2535_253593

theorem problem_solution (x : ℝ) : x * 120 = 346 → x = 346 / 120 := by
  sorry

end problem_solution_l2535_253593


namespace bbq_ice_per_person_l2535_253562

/-- Given the conditions of Chad's BBQ, prove that the amount of ice needed per person is 2 pounds. -/
theorem bbq_ice_per_person (people : ℕ) (pack_price : ℚ) (pack_size : ℕ) (total_spent : ℚ) :
  people = 15 →
  pack_price = 3 →
  pack_size = 10 →
  total_spent = 9 →
  (total_spent / pack_price * pack_size) / people = 2 := by
  sorry

#check bbq_ice_per_person

end bbq_ice_per_person_l2535_253562


namespace water_left_over_l2535_253545

theorem water_left_over (players : ℕ) (initial_water : ℕ) (water_per_player : ℕ) (spilled_water : ℕ) :
  players = 30 →
  initial_water = 8000 →
  water_per_player = 200 →
  spilled_water = 250 →
  initial_water - (players * water_per_player + spilled_water) = 1750 :=
by
  sorry

end water_left_over_l2535_253545


namespace longest_pole_in_room_l2535_253564

theorem longest_pole_in_room (length width height : ℝ) 
  (h_length : length = 12)
  (h_width : width = 8)
  (h_height : height = 9) :
  Real.sqrt (length^2 + width^2 + height^2) = 17 := by
  sorry

end longest_pole_in_room_l2535_253564


namespace greatest_real_part_of_sixth_power_l2535_253581

theorem greatest_real_part_of_sixth_power : 
  let z₁ : ℂ := -3
  let z₂ : ℂ := -Real.sqrt 6 + Complex.I
  let z₃ : ℂ := -Real.sqrt 3 + (Real.sqrt 3 : ℝ) * Complex.I
  let z₄ : ℂ := -1 + (Real.sqrt 6 : ℝ) * Complex.I
  let z₅ : ℂ := 2 * Complex.I
  Complex.re (z₁^6) > Complex.re (z₂^6) ∧
  Complex.re (z₁^6) > Complex.re (z₃^6) ∧
  Complex.re (z₁^6) > Complex.re (z₄^6) ∧
  Complex.re (z₁^6) > Complex.re (z₅^6) :=
by
  sorry

end greatest_real_part_of_sixth_power_l2535_253581


namespace smallest_number_satisfying_conditions_l2535_253555

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_number_satisfying_conditions : 
  (∀ n : ℕ, n < 70 → ¬(is_multiple_of n 7 ∧ is_multiple_of n 5 ∧ is_prime (n + 9))) ∧ 
  (is_multiple_of 70 7 ∧ is_multiple_of 70 5 ∧ is_prime (70 + 9)) :=
sorry

end smallest_number_satisfying_conditions_l2535_253555


namespace consecutive_integers_equation_l2535_253566

theorem consecutive_integers_equation (x y z n : ℤ) : 
  x = y + 1 → 
  y = z + 1 → 
  x > y → 
  y > z → 
  z = 3 → 
  2*x + 3*y + 3*z = 5*y + n → 
  n = 11 := by sorry

end consecutive_integers_equation_l2535_253566


namespace rachels_father_age_at_25_is_60_l2535_253582

/-- Calculates the age of Rachel's father when Rachel is 25 years old -/
def rachels_father_age_at_25 (rachel_current_age : ℕ) (grandfather_age_multiplier : ℕ) (father_age_difference : ℕ) : ℕ :=
  let grandfather_age := rachel_current_age * grandfather_age_multiplier
  let mother_age := grandfather_age / 2
  let father_current_age := mother_age + father_age_difference
  let years_until_25 := 25 - rachel_current_age
  father_current_age + years_until_25

/-- Theorem stating that Rachel's father will be 60 years old when Rachel is 25 -/
theorem rachels_father_age_at_25_is_60 :
  rachels_father_age_at_25 12 7 5 = 60 := by
  sorry

#eval rachels_father_age_at_25 12 7 5

end rachels_father_age_at_25_is_60_l2535_253582


namespace polynomial_roots_l2535_253530

/-- The polynomial x^3 - 7x^2 + 11x + 13 -/
def f (x : ℝ) := x^3 - 7*x^2 + 11*x + 13

/-- The set of roots of the polynomial -/
def roots : Set ℝ := {2, 6, -1}

theorem polynomial_roots :
  (∀ x ∈ roots, f x = 0) ∧
  (∀ x : ℝ, f x = 0 → x ∈ roots) :=
sorry

end polynomial_roots_l2535_253530


namespace division_of_decimals_l2535_253556

theorem division_of_decimals : (0.36 : ℝ) / (0.004 : ℝ) = 90 := by sorry

end division_of_decimals_l2535_253556


namespace number_of_factors_of_b_power_n_l2535_253580

def b : ℕ := 6
def n : ℕ := 15

theorem number_of_factors_of_b_power_n : 
  b ≤ 15 → n ≤ 15 → (Nat.factors (b^n)).length + 1 = 256 := by
  sorry

end number_of_factors_of_b_power_n_l2535_253580


namespace economics_test_absentees_l2535_253528

theorem economics_test_absentees (total_students : ℕ) (q1_correct : ℕ) (q2_correct : ℕ) (both_correct : ℕ) 
  (h1 : total_students = 29)
  (h2 : q1_correct = 19)
  (h3 : q2_correct = 24)
  (h4 : both_correct = 19) :
  total_students - (q1_correct + q2_correct - both_correct) = 5 := by
  sorry


end economics_test_absentees_l2535_253528


namespace percentage_of_75_to_125_l2535_253563

theorem percentage_of_75_to_125 : ∀ (x : ℝ), x = (75 : ℝ) / (125 : ℝ) * 100 → x = 60 :=
by
  sorry

end percentage_of_75_to_125_l2535_253563


namespace square_perimeter_diagonal_ratio_l2535_253588

theorem square_perimeter_diagonal_ratio (P₁ P₂ d₁ d₂ : ℝ) :
  P₁ > 0 ∧ P₂ > 0 ∧ d₁ > 0 ∧ d₂ > 0 ∧ 
  (P₂ / P₁ = 11) ∧
  (P₁ = 4 * (d₁ / Real.sqrt 2)) ∧
  (P₂ = 4 * (d₂ / Real.sqrt 2)) →
  d₂ / d₁ = 11 := by
sorry

end square_perimeter_diagonal_ratio_l2535_253588


namespace water_height_in_aquarium_l2535_253554

/-- Proves that the height of water in an aquarium with given dimensions and volume of water is 10 cm. -/
theorem water_height_in_aquarium :
  let aquarium_length : ℝ := 50
  let aquarium_breadth : ℝ := 20
  let aquarium_height : ℝ := 40
  let water_volume : ℝ := 10000  -- 10 litres * 1000 cm³/litre
  let water_height : ℝ := water_volume / (aquarium_length * aquarium_breadth)
  water_height = 10 := by sorry

end water_height_in_aquarium_l2535_253554


namespace my_matrix_is_projection_l2535_253526

def projection_matrix (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * A = A

def my_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![9/25, 18/45],
    ![12/25, 27/45]]

theorem my_matrix_is_projection : projection_matrix my_matrix := by
  sorry

end my_matrix_is_projection_l2535_253526


namespace inequality_solutions_l2535_253504

theorem inequality_solutions :
  (∀ x : ℝ, x^2 - 5*x + 5 > 0 ↔ (x > (5 + Real.sqrt 5) / 2 ∨ x < (5 - Real.sqrt 5) / 2)) ∧
  (∀ x : ℝ, -2*x^2 + x - 3 < 0) := by
sorry

end inequality_solutions_l2535_253504


namespace binary_to_octal_conversion_l2535_253516

theorem binary_to_octal_conversion : 
  (1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 
  (1 * 8^2 + 1 * 8^1 + 5 * 8^0) :=
by sorry

end binary_to_octal_conversion_l2535_253516


namespace otimes_example_l2535_253523

-- Define the ⊗ operation
def otimes (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem otimes_example : otimes 4 (otimes 2 (-1)) = 7 := by sorry

end otimes_example_l2535_253523


namespace initial_amount_proof_l2535_253538

/-- 
Theorem: If an amount increases by 1/8th of itself each year for two years 
and results in 82265.625, then the initial amount was 65000.
-/
theorem initial_amount_proof (initial_amount : ℚ) : 
  (initial_amount * (9/8)^2 = 82265.625) → initial_amount = 65000 := by
  sorry

end initial_amount_proof_l2535_253538


namespace hydra_disconnect_l2535_253536

/-- A graph representing a hydra -/
structure Hydra where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  vertex_count : vertices.card = 100

/-- Invert a vertex in the hydra -/
def invert_vertex (H : Hydra) (v : Nat) : Hydra :=
  sorry

/-- Check if the hydra is disconnected -/
def is_disconnected (H : Hydra) : Prop :=
  sorry

/-- Main theorem: Any 100-vertex hydra can be disconnected in at most 10 inversions -/
theorem hydra_disconnect (H : Hydra) :
  ∃ (inversions : List Nat), inversions.length ≤ 10 ∧
    is_disconnected (inversions.foldl invert_vertex H) :=
  sorry

end hydra_disconnect_l2535_253536


namespace irrational_and_rational_numbers_l2535_253550

theorem irrational_and_rational_numbers : ∃ (x : ℝ), 
  (Irrational (-Real.sqrt 5)) ∧ 
  (¬ Irrational (Real.sqrt 4)) ∧ 
  (¬ Irrational (2 / 3)) ∧ 
  (¬ Irrational 0) := by
  sorry

end irrational_and_rational_numbers_l2535_253550


namespace intersection_line_through_origin_l2535_253533

/-- Given two lines l₁ and l₂ in the plane, prove that the line passing through
    their intersection point and the origin has the equation x - 10y = 0. -/
theorem intersection_line_through_origin
  (l₁ : Set (ℝ × ℝ))
  (l₂ : Set (ℝ × ℝ))
  (h₁ : l₁ = {(x, y) | 2 * x + y = 3})
  (h₂ : l₂ = {(x, y) | x + 4 * y = 2})
  (P : ℝ × ℝ)
  (hP : P ∈ l₁ ∧ P ∈ l₂)
  (l : Set (ℝ × ℝ))
  (hl : l = {(x, y) | ∃ t : ℝ, x = t * P.1 ∧ y = t * P.2}) :
  l = {(x, y) | x - 10 * y = 0} :=
sorry

end intersection_line_through_origin_l2535_253533


namespace floor_of_3_999_l2535_253599

theorem floor_of_3_999 : ⌊(3.999 : ℝ)⌋ = 3 := by sorry

end floor_of_3_999_l2535_253599


namespace log_simplification_l2535_253543

-- Define variables
variable (a b c d x y : ℝ)
-- Assume all variables are positive to ensure logarithms are defined
variable (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hx : x > 0) (hy : y > 0)

-- Define the theorem
theorem log_simplification :
  Real.log (2*a/(3*b)) + Real.log (5*b/(4*c)) + Real.log (6*c/(7*d)) - Real.log (20*a*y/(21*d*x)) = Real.log (3*x/(4*y)) :=
by sorry

end log_simplification_l2535_253543


namespace game_draw_fraction_l2535_253542

theorem game_draw_fraction (ben_win : ℚ) (tom_win : ℚ) (draw : ℚ) : 
  ben_win = 4/9 → tom_win = 1/3 → draw = 1 - (ben_win + tom_win) → draw = 2/9 := by
  sorry

end game_draw_fraction_l2535_253542


namespace symmetric_point_theorem_l2535_253532

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Finds the point symmetric to a given point with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- The theorem stating that the point symmetric to (-3, -4, 5) with respect to the xOz plane is (-3, 4, 5) -/
theorem symmetric_point_theorem :
  let A : Point3D := { x := -3, y := -4, z := 5 }
  symmetricPointXOZ A = { x := -3, y := 4, z := 5 } := by
  sorry


end symmetric_point_theorem_l2535_253532


namespace base_nine_solution_l2535_253585

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem base_nine_solution :
  ∃! b : Nat, b > 0 ∧ 
    to_decimal [1, 7, 2] b + to_decimal [1, 4, 5] b = to_decimal [3, 2, 7] b :=
by sorry

end base_nine_solution_l2535_253585


namespace complex_number_properties_l2535_253518

theorem complex_number_properties (z : ℂ) (h : Complex.abs z ^ 2 + 2 * z - Complex.I * 2 = 0) :
  z = -1 + Complex.I ∧ Complex.abs z + Complex.abs (z + 3 * Complex.I) > Complex.abs (2 * z + 3 * Complex.I) := by
  sorry

end complex_number_properties_l2535_253518


namespace always_quadratic_in_x_l2535_253510

/-- A quadratic equation in x is of the form ax² + bx + c = 0 where a ≠ 0 -/
def is_quadratic_in_x (a b c : ℝ) : Prop := a ≠ 0

/-- The equation (m²+1)x² - mx - 3 = 0 is quadratic in x for all real m -/
theorem always_quadratic_in_x (m : ℝ) : 
  is_quadratic_in_x (m^2 + 1) (-m) (-3) := by sorry

end always_quadratic_in_x_l2535_253510


namespace plywood_perimeter_l2535_253594

theorem plywood_perimeter (length width perimeter : ℝ) : 
  length = 6 → width = 5 → perimeter = 2 * (length + width) → perimeter = 22 := by
  sorry

end plywood_perimeter_l2535_253594


namespace container_volume_ratio_l2535_253572

theorem container_volume_ratio : 
  ∀ (A B : ℝ),  -- A and B are the volumes of the first and second containers
  A > 0 → B > 0 →  -- Both volumes are positive
  (4/5 * A - 1/5 * A) = 2/3 * B →  -- Amount poured equals 2/3 of second container
  A / B = 10/9 := by
  sorry

end container_volume_ratio_l2535_253572


namespace sales_function_properties_l2535_253551

def f (x : ℝ) : ℝ := x^2 - 7*x + 14

theorem sales_function_properties :
  (∃ (a b : ℝ), a < b ∧ 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y) ∧
    (∀ x y, b ≤ x ∧ x < y → f x ≤ f y)) ∧
  f 1 = 8 ∧
  f 3 = 2 := by sorry

end sales_function_properties_l2535_253551


namespace square_difference_of_integers_l2535_253579

theorem square_difference_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 50) 
  (diff_eq : x - y = 12) : 
  x^2 - y^2 = 600 := by
  sorry

end square_difference_of_integers_l2535_253579


namespace hidden_dots_sum_l2535_253505

/-- Represents a standard six-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sum of all numbers on a standard die -/
def DieSum : ℕ := Finset.sum StandardDie id

/-- The number of dice in the stack -/
def NumDice : ℕ := 4

/-- The visible numbers on the stack -/
def VisibleNumbers : Finset ℕ := {1, 2, 3, 5, 6}

/-- The sum of visible numbers -/
def VisibleSum : ℕ := Finset.sum VisibleNumbers id

theorem hidden_dots_sum :
  NumDice * DieSum - VisibleSum = 67 := by sorry

end hidden_dots_sum_l2535_253505


namespace tangent_and_inequality_l2535_253517

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.log (x + 1) - a

theorem tangent_and_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀ = 0 ∧ f a x₀ = 0 ∧ (deriv (f a)) x₀ = 0) →
  (∀ x t : ℝ, x > t ∧ t ≥ 0 → Real.exp (x - t) + Real.log (t + 1) > Real.log (x + 1) + 1) ∧
  (f a = f 1) :=
by sorry

end tangent_and_inequality_l2535_253517


namespace cubic_root_sum_cubes_l2535_253559

theorem cubic_root_sum_cubes (p q r : ℝ) : 
  (p^3 - 2*p^2 + 3*p - 1 = 0) ∧ 
  (q^3 - 2*q^2 + 3*q - 1 = 0) ∧ 
  (r^3 - 2*r^2 + 3*r - 1 = 0) →
  p^3 + q^3 + r^3 = -7 :=
by sorry

end cubic_root_sum_cubes_l2535_253559


namespace min_velocity_increase_is_6_l2535_253587

/-- Represents a car with its velocity -/
structure Car where
  velocity : ℝ

/-- Represents the road scenario -/
structure RoadScenario where
  carA : Car
  carB : Car
  carC : Car
  initialDistanceAB : ℝ
  initialDistanceAC : ℝ

/-- Calculates the minimum velocity increase needed for car A -/
def minVelocityIncrease (scenario : RoadScenario) : ℝ :=
  sorry

/-- Theorem stating the minimum velocity increase for the given scenario -/
theorem min_velocity_increase_is_6 (scenario : RoadScenario) 
  (h1 : scenario.carA.velocity > scenario.carB.velocity)
  (h2 : scenario.initialDistanceAB = 50)
  (h3 : scenario.initialDistanceAC = 300)
  (h4 : scenario.carB.velocity = 50)
  (h5 : scenario.carC.velocity = 70)
  (h6 : scenario.carA.velocity = 68) :
  minVelocityIncrease scenario = 6 :=
sorry

end min_velocity_increase_is_6_l2535_253587


namespace smallest_k_for_product_sign_change_l2535_253557

def sequence_a (n : ℕ) : ℚ :=
  15 - 2/3 * (n - 1)

theorem smallest_k_for_product_sign_change :
  let a := sequence_a
  (∀ n : ℕ, n ≥ 1 → 3 * a (n + 1) = 3 * a n - 2) →
  (∃ k : ℕ, k > 0 ∧ a k * a (k + 1) < 0) →
  (∀ j : ℕ, 0 < j → j < 23 → a j * a (j + 1) ≥ 0) →
  a 23 * a 24 < 0 :=
by sorry

end smallest_k_for_product_sign_change_l2535_253557


namespace total_swim_distance_l2535_253512

/-- The total distance Molly swam on Saturday in meters -/
def saturday_distance : ℕ := 400

/-- The total distance Molly swam on Sunday in meters -/
def sunday_distance : ℕ := 300

/-- The theorem states that the total distance Molly swam in all four pools
    is equal to the sum of the distances she swam on Saturday and Sunday -/
theorem total_swim_distance :
  saturday_distance + sunday_distance = 700 := by
  sorry

end total_swim_distance_l2535_253512


namespace solve_for_m_l2535_253552

theorem solve_for_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = 6) : m = 7 := by
  sorry

end solve_for_m_l2535_253552


namespace plane_stops_at_20_seconds_stop_time_unique_l2535_253548

/-- The distance function representing the plane's movement after landing -/
def s (t : ℝ) : ℝ := -1.5 * t^2 + 60 * t

/-- The time at which the plane stops -/
def stop_time : ℝ := 20

/-- Theorem stating that the plane stops at 20 seconds -/
theorem plane_stops_at_20_seconds :
  (∀ t : ℝ, t ≥ 0 → s t ≤ s stop_time) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ t, |t - stop_time| < δ → s t < s stop_time) := by
  sorry

/-- Corollary: The stop time is unique -/
theorem stop_time_unique (t : ℝ) :
  (∀ τ : ℝ, τ ≥ 0 → s τ ≤ s t) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ τ, |τ - t| < δ → s τ < s t) →
  t = stop_time := by
  sorry

end plane_stops_at_20_seconds_stop_time_unique_l2535_253548


namespace constant_sum_product_l2535_253501

theorem constant_sum_product (n : Nat) (h : n = 15) : 
  ∃ k : Nat, ∀ (operations : List (Nat × Nat)), 
    operations.length = n - 1 → 
    (∀ (x y : Nat), (x, y) ∈ operations → x ≤ n ∧ y ≤ n) →
    (List.foldl (λ acc (x, y) => acc + x * y * (x + y)) 0 operations) = k ∧ k = 49140 := by
  sorry

end constant_sum_product_l2535_253501


namespace equal_chicken_wing_distribution_l2535_253596

theorem equal_chicken_wing_distribution 
  (num_friends : ℕ)
  (pre_cooked_wings : ℕ)
  (additional_wings : ℕ)
  (h1 : num_friends = 4)
  (h2 : pre_cooked_wings = 9)
  (h3 : additional_wings = 7) :
  (pre_cooked_wings + additional_wings) / num_friends = 4 :=
by sorry

end equal_chicken_wing_distribution_l2535_253596


namespace star_difference_l2535_253586

-- Define the ⭐ operation
def star (x y : ℝ) : ℝ := x^2 * y - 3 * x + y

-- Theorem statement
theorem star_difference : star 3 5 - star 5 3 = -22 := by
  sorry

end star_difference_l2535_253586


namespace contrapositive_equivalence_l2535_253544

theorem contrapositive_equivalence (a b c d : ℝ) :
  ((a = b ∧ c = d) → (a + c = b + d)) ↔ ((a + c ≠ b + d) → (a ≠ b ∨ c ≠ d)) := by
  sorry

end contrapositive_equivalence_l2535_253544


namespace polynomial_factorization_l2535_253539

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 5) * (x^2 + 6*x + 2) := by
  sorry

end polynomial_factorization_l2535_253539


namespace average_score_is_1_9_l2535_253506

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_3_percent : ℚ
  score_2_percent : ℚ
  score_1_percent : ℚ
  score_0_percent : ℚ

/-- Calculates the average score of a class given its score distribution -/
def average_score (sd : ScoreDistribution) : ℚ :=
  (3 * sd.score_3_percent + 2 * sd.score_2_percent + sd.score_1_percent) * sd.total_students / 100

/-- The theorem stating that the average score for the given distribution is 1.9 -/
theorem average_score_is_1_9 (sd : ScoreDistribution)
  (h1 : sd.total_students = 30)
  (h2 : sd.score_3_percent = 30)
  (h3 : sd.score_2_percent = 40)
  (h4 : sd.score_1_percent = 20)
  (h5 : sd.score_0_percent = 10) :
  average_score sd = 19/10 := by sorry

end average_score_is_1_9_l2535_253506


namespace circle_arrangement_impossibility_l2535_253535

theorem circle_arrangement_impossibility :
  ¬ ∃ (a : Fin 7 → ℕ),
    (∀ i : Fin 7, ∃ j : Fin 7, (a j + a ((j + 1) % 7) + a ((j + 2) % 7) = i + 1)) ∧
    (∀ i j : Fin 7, i ≠ j → 
      (a i + a ((i + 1) % 7) + a ((i + 2) % 7)) ≠ (a j + a ((j + 1) % 7) + a ((j + 2) % 7))) :=
by sorry

end circle_arrangement_impossibility_l2535_253535


namespace midpoint_ratio_range_l2535_253513

-- Define the lines and points
def line1 (x y : ℝ) : Prop := x + 3 * y - 2 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y + 6 = 0

-- Define the midpoint condition
def is_midpoint (x₀ y₀ x_p y_p x_q y_q : ℝ) : Prop :=
  x₀ = (x_p + x_q) / 2 ∧ y₀ = (y_p + y_q) / 2

-- State the theorem
theorem midpoint_ratio_range (x₀ y₀ x_p y_p x_q y_q : ℝ) :
  line1 x_p y_p →
  line2 x_q y_q →
  is_midpoint x₀ y₀ x_p y_p x_q y_q →
  y₀ < x₀ + 2 →
  (y₀ / x₀ < -1/3 ∨ y₀ / x₀ > 0) :=
by sorry

end midpoint_ratio_range_l2535_253513


namespace largest_last_digit_l2535_253574

/-- A string of digits satisfying the problem conditions -/
def ValidString : Type := 
  {s : List Nat // s.length = 2007 ∧ s.head! = 2 ∧ 
    ∀ i, i < 2006 → (s.get! i * 10 + s.get! (i+1)) % 23 = 0 ∨ 
                     (s.get! i * 10 + s.get! (i+1)) % 37 = 0}

/-- The theorem stating the largest possible last digit -/
theorem largest_last_digit (s : ValidString) : s.val.getLast! ≤ 9 :=
sorry

end largest_last_digit_l2535_253574


namespace determine_set_N_l2535_253576

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define subset M
def M : Set Nat := {1, 4}

-- Define the theorem
theorem determine_set_N (N : Set Nat) 
  (h1 : N ⊆ U)
  (h2 : M ∩ N = {1})
  (h3 : N ∩ (U \ M) = {3, 5}) :
  N = {1, 3, 5} := by
  sorry

end determine_set_N_l2535_253576


namespace arithmetic_sequence_2011_l2535_253553

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

/-- The problem statement -/
theorem arithmetic_sequence_2011 :
  arithmeticSequenceTerm 1 3 671 = 2011 := by
  sorry

end arithmetic_sequence_2011_l2535_253553


namespace friends_at_reception_l2535_253527

/-- Calculates the number of friends attending a wedding reception --/
theorem friends_at_reception (total_guests : ℕ) (bride_couples : ℕ) (groom_couples : ℕ) : 
  total_guests - 2 * (bride_couples + groom_couples) = 100 :=
by
  sorry

#check friends_at_reception 180 20 20

end friends_at_reception_l2535_253527


namespace track_circumference_l2535_253520

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (v1 v2 t : ℝ) (h1 : v1 = 4.5) (h2 : v2 = 3.75) (h3 : t = 5.28 / 60) :
  v1 * t + v2 * t = 0.726 := by
  sorry

end track_circumference_l2535_253520


namespace percentage_problem_l2535_253540

theorem percentage_problem (P : ℝ) (number : ℝ) : 
  number = 40 →
  P = (0.5 * number) + 10 →
  P = 30 :=
by sorry

end percentage_problem_l2535_253540


namespace sandy_comic_books_l2535_253511

/-- Proves that Sandy bought 6 comic books given the initial conditions -/
theorem sandy_comic_books :
  let initial_books : ℕ := 14
  let sold_books : ℕ := initial_books / 2
  let current_books : ℕ := 13
  let bought_books : ℕ := current_books - (initial_books - sold_books)
  bought_books = 6 := by
  sorry

end sandy_comic_books_l2535_253511


namespace set_A_determination_l2535_253507

universe u

def U : Set ℕ := {1, 2, 3, 4}

theorem set_A_determination (A : Set ℕ) 
  (h1 : A ⊆ U)
  (h2 : A ∩ {1, 2, 3} = {2})
  (h3 : A ∪ {1, 2, 3} = U) :
  A = {2, 4} := by
sorry


end set_A_determination_l2535_253507


namespace eggs_per_box_l2535_253584

/-- Given that Maria has 3 boxes of eggs and a total of 21 eggs, 
    prove that each box contains 7 eggs. -/
theorem eggs_per_box (total_eggs : ℕ) (num_boxes : ℕ) 
  (h1 : total_eggs = 21) (h2 : num_boxes = 3) : 
  total_eggs / num_boxes = 7 := by
  sorry

end eggs_per_box_l2535_253584


namespace income_growth_equation_correct_l2535_253561

/-- Represents the growth of per capita disposable income in China from 2020 to 2022 -/
def income_growth (x : ℝ) : Prop :=
  let income_2020 : ℝ := 3.2  -- in ten thousand yuan
  let income_2022 : ℝ := 3.7  -- in ten thousand yuan
  let years : ℕ := 2
  income_2020 * (1 + x) ^ years = income_2022

/-- Theorem stating that the equation correctly represents the income growth -/
theorem income_growth_equation_correct :
  ∃ x : ℝ, income_growth x := by
  sorry

end income_growth_equation_correct_l2535_253561


namespace real_solutions_quadratic_l2535_253571

theorem real_solutions_quadratic (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 - 3 * x * y + x + 8 = 0) ↔ x ≤ -4 ∨ x ≥ 8 := by
  sorry

end real_solutions_quadratic_l2535_253571


namespace max_value_trig_product_l2535_253573

theorem max_value_trig_product (x y z : ℝ) :
  (Real.sin (2*x) + Real.sin y + Real.sin (3*z)) * 
  (Real.cos (2*x) + Real.cos y + Real.cos (3*z)) ≤ 4.5 := by
  sorry

end max_value_trig_product_l2535_253573


namespace snow_probability_first_week_l2535_253558

def probability_of_snow (days : ℕ) (daily_prob : ℚ) : ℚ :=
  1 - (1 - daily_prob) ^ days

theorem snow_probability_first_week :
  let prob_first_four := probability_of_snow 4 (1/4)
  let prob_next_three := probability_of_snow 3 (1/3)
  1 - (1 - prob_first_four) * (1 - prob_next_three) = 29/32 := by
  sorry

end snow_probability_first_week_l2535_253558


namespace complex_norm_problem_l2535_253529

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 12)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z - w) = 7) :
  Complex.abs w = Real.sqrt 36.75 :=
sorry

end complex_norm_problem_l2535_253529


namespace remainder_of_sequence_sum_l2535_253514

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem remainder_of_sequence_sum :
  ∃ n : ℕ, 
    arithmetic_sequence 1 6 n = 403 ∧ 
    sequence_sum 1 6 n % 6 = 2 := by
  sorry

end remainder_of_sequence_sum_l2535_253514


namespace exam_score_problem_l2535_253569

theorem exam_score_problem (scores : List ℝ) (avg : ℝ) : 
  scores.length = 4 →
  scores = [80, 90, 100, 110] →
  avg = 96 →
  (scores.sum + (5 * avg - scores.sum)) / 5 = avg →
  5 * avg - scores.sum = 100 := by
sorry

end exam_score_problem_l2535_253569


namespace max_a_when_a_squared_plus_100a_prime_l2535_253598

theorem max_a_when_a_squared_plus_100a_prime (a : ℕ+) :
  Nat.Prime (a^2 + 100*a) → a ≤ 1 := by
  sorry

end max_a_when_a_squared_plus_100a_prime_l2535_253598


namespace sine_symmetry_axis_symmetric_angles_sqrt_cos_minus_one_even_l2535_253522

open Real

-- Statement 2
theorem sine_symmetry_axis (k : ℤ) :
  ∀ x : ℝ, sin x = sin (π - x + (k * 2 * π)) := by sorry

-- Statement 3
theorem symmetric_angles (α β : ℝ) (k : ℤ) :
  (∀ x : ℝ, sin (α + x) = sin (β - x)) →
  α + β = (2 * k - 1) * π := by sorry

-- Statement 5
theorem sqrt_cos_minus_one_even :
  ∀ x : ℝ, sqrt (cos x - 1) = sqrt (cos (-x) - 1) := by sorry

end sine_symmetry_axis_symmetric_angles_sqrt_cos_minus_one_even_l2535_253522


namespace seaplane_speed_l2535_253508

theorem seaplane_speed (v : ℝ) (h1 : v > 0) : 
  (2 : ℝ) / ((1 / v) + (1 / 72)) = 91 → v = 6552 / 53 := by
  sorry

end seaplane_speed_l2535_253508


namespace calculator_time_saved_l2535_253547

/-- Proves that using a calculator saves 150 minutes for Matt's math homework -/
theorem calculator_time_saved 
  (time_with_calc : ℕ) 
  (time_without_calc : ℕ) 
  (num_problems : ℕ) 
  (h1 : time_with_calc = 3)
  (h2 : time_without_calc = 8)
  (h3 : num_problems = 30) :
  time_without_calc * num_problems - time_with_calc * num_problems = 150 :=
by sorry

end calculator_time_saved_l2535_253547


namespace van_distance_proof_l2535_253595

theorem van_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 32 →
  (initial_time * 3 / 2) * new_speed = 288 := by
  sorry

end van_distance_proof_l2535_253595


namespace max_floors_theorem_l2535_253565

/-- Represents a building with elevators and floors -/
structure Building where
  num_elevators : ℕ
  num_floors : ℕ
  stops_per_elevator : ℕ
  all_pairs_connected : Bool

/-- The maximum number of floors possible for a building with given constraints -/
def max_floors (b : Building) : ℕ :=
  sorry

/-- Theorem stating that for a building with 7 elevators, each stopping on 6 floors,
    and all pairs of floors connected, the maximum number of floors is 14 -/
theorem max_floors_theorem (b : Building) 
  (h1 : b.num_elevators = 7)
  (h2 : b.stops_per_elevator = 6)
  (h3 : b.all_pairs_connected = true) :
  max_floors b = 14 := by
  sorry

end max_floors_theorem_l2535_253565


namespace third_number_proof_l2535_253534

theorem third_number_proof (a b c : ℝ) : 
  a = 6 → b = 16 → (a + b + c) / 3 = 13 → c = 17 := by
  sorry

end third_number_proof_l2535_253534


namespace quadratic_equation_coefficients_l2535_253577

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ), 
  (∀ x, 9 * x^2 = 4 * (3 * x - 1)) →
  (∀ x, a * x^2 + b * x + c = 0) →
  a = 9 ∧ b = -12 ∧ c = 4 := by
sorry

end quadratic_equation_coefficients_l2535_253577


namespace problem_solution_l2535_253560

theorem problem_solution (x : ℝ) : (0.25 * x = 0.15 * 1500 - 30) → x = 780 := by
  sorry

end problem_solution_l2535_253560


namespace white_to_black_stone_ratio_l2535_253592

theorem white_to_black_stone_ratio :
  ∀ (total_stones white_stones black_stones : ℕ),
    total_stones = 100 →
    white_stones = 60 →
    black_stones = total_stones - white_stones →
    white_stones > black_stones →
    (white_stones : ℚ) / (black_stones : ℚ) = 3 / 2 := by
  sorry

end white_to_black_stone_ratio_l2535_253592


namespace base3_to_base10_equiv_l2535_253570

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number -/
def base3Number : List Nat := [1, 2, 2, 0, 1]

theorem base3_to_base10_equiv : base3ToBase10 base3Number = 106 := by
  sorry

end base3_to_base10_equiv_l2535_253570


namespace positive_real_properties_l2535_253591

theorem positive_real_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b - a * b = 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 4 * y - x * y = 0 ∧ x + 4 * y < a + 4 * b) →
  (a + 2 * b ≥ 6 + 4 * Real.sqrt 2) ∧
  (16 / a^2 + 1 / b^2 ≥ 1 / 2) :=
by sorry

end positive_real_properties_l2535_253591


namespace equation_solution_l2535_253531

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (21 / (x₁^2 - 9) - 3 / (x₁ - 3) = 2) ∧ 
                 (21 / (x₂^2 - 9) - 3 / (x₂ - 3) = 2) ∧ 
                 (abs (x₁ - 4.695) < 0.001) ∧ 
                 (abs (x₂ + 3.195) < 0.001) := by
  sorry

end equation_solution_l2535_253531


namespace geometric_sequence_sum_l2535_253519

theorem geometric_sequence_sum (a : ℝ) : 
  (a + 2*a + 4*a + 8*a = 1) →  -- Sum of first 4 terms equals 1
  (a + 2*a + 4*a + 8*a + 16*a + 32*a + 64*a + 128*a = 17) :=  -- Sum of first 8 terms equals 17
by
  sorry

end geometric_sequence_sum_l2535_253519
