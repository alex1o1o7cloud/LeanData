import Mathlib

namespace vectors_form_basis_l3786_378682

def vector_a : Fin 2 → ℝ := λ i => if i = 0 then 2 else -3
def vector_b : Fin 2 → ℝ := λ i => if i = 0 then 6 else 9

def is_basis (v w : Fin 2 → ℝ) : Prop :=
  LinearIndependent ℝ (![v, w]) ∧ Submodule.span ℝ {v, w} = ⊤

theorem vectors_form_basis : is_basis vector_a vector_b := by sorry

end vectors_form_basis_l3786_378682


namespace least_subtraction_for_divisibility_by_ten_l3786_378690

theorem least_subtraction_for_divisibility_by_ten (n : ℕ) (h : n = 427398) :
  ∃ (k : ℕ), k = 8 ∧ (n - k) % 10 = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % 10 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_by_ten_l3786_378690


namespace bottles_per_case_l3786_378663

/-- Represents the number of bottles produced in a day -/
def total_bottles : ℕ := 120000

/-- Represents the number of cases required for one day's production -/
def total_cases : ℕ := 10000

/-- Theorem stating that the number of bottles per case is 12 -/
theorem bottles_per_case :
  total_bottles / total_cases = 12 := by
  sorry

end bottles_per_case_l3786_378663


namespace set_equality_l3786_378666

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the set we want to prove equal to ℂᵤ(M ∪ N)
def target_set : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem set_equality : target_set = (M ∪ N)ᶜ := by sorry

end set_equality_l3786_378666


namespace complex_difference_magnitude_l3786_378667

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1)
  (h2 : Complex.abs z₂ = 1)
  (h3 : Complex.abs (z₁ + z₂) = Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 1 := by
  sorry

end complex_difference_magnitude_l3786_378667


namespace add_specific_reals_l3786_378636

theorem add_specific_reals : 1.25 + 47.863 = 49.113 := by
  sorry

end add_specific_reals_l3786_378636


namespace probability_above_parabola_l3786_378612

/-- The type of single-digit positive integers -/
def SingleDigitPos := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- The condition for a point (a,b) to be above the parabola y = ax^2 + bx for all x -/
def IsAboveParabola (a b : SingleDigitPos) : Prop :=
  ∀ x : ℝ, (b : ℝ) > (a : ℝ) * x^2 + (b : ℝ) * x

/-- The number of valid (a,b) pairs -/
def NumValidPairs : ℕ := 72

/-- The total number of possible (a,b) pairs -/
def TotalPairs : ℕ := 81

/-- The main theorem: the probability of (a,b) being above the parabola is 8/9 -/
theorem probability_above_parabola :
  (NumValidPairs : ℚ) / (TotalPairs : ℚ) = 8 / 9 :=
sorry

end probability_above_parabola_l3786_378612


namespace sum_of_squared_distances_l3786_378680

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- A point on the circumcircle of a triangle -/
def point_on_circumcircle (t : Triangle) : ℝ × ℝ := sorry

/-- Squared distance between two points -/
def squared_distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem sum_of_squared_distances (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  let R := circumradius t
  let P := point_on_circumcircle t
  squared_distance P t.A + squared_distance P t.B + squared_distance P t.C + squared_distance P H = 8 * R^2 := by
  sorry

end sum_of_squared_distances_l3786_378680


namespace max_value_problem_1_l3786_378686

theorem max_value_problem_1 (x : ℝ) (h : x < 5/4) :
  ∃ (y : ℝ), y = 4*x - 2 + 1/(4*x - 5) ∧ y ≤ 1 ∧ (∀ (z : ℝ), z = 4*x - 2 + 1/(4*x - 5) → z ≤ y) :=
sorry

end max_value_problem_1_l3786_378686


namespace angle_equivalence_l3786_378662

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (h1 : A.2 = B.2 ∧ B.2 = C.2 ∧ C.2 = D.2)  -- A, B, C, D are on the same line
variable (h2 : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1)  -- A, B, C, D are in that order
variable (h3 : dist A B = dist C D)  -- AB = CD
variable (h4 : E.2 ≠ A.2)  -- E is off the line
variable (h5 : dist C E = dist D E)  -- CE = DE

-- Define the angle function
def angle (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem angle_equivalence :
  angle C E D = 2 * angle A E B ↔ dist A C = dist E C :=
sorry

end angle_equivalence_l3786_378662


namespace solve_for_s_l3786_378617

theorem solve_for_s (s t : ℚ) 
  (eq1 : 8 * s + 7 * t = 160) 
  (eq2 : s = t - 3) : 
  s = 139 / 15 := by
sorry

end solve_for_s_l3786_378617


namespace church_cookies_total_l3786_378645

/-- Calculates the total number of cookies baked by church members -/
theorem church_cookies_total (members : ℕ) (sheets_per_member : ℕ) (cookies_per_sheet : ℕ) : 
  members = 100 → sheets_per_member = 10 → cookies_per_sheet = 16 → 
  members * sheets_per_member * cookies_per_sheet = 16000 := by
  sorry

end church_cookies_total_l3786_378645


namespace regular_polygon_perimeter_l3786_378694

/-- A regular polygon with side length 8 units and exterior angle 90 degrees has a perimeter of 32 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 8 ∧ 
  exterior_angle = 90 ∧ 
  (360 : ℝ) / n = exterior_angle → 
  n * side_length = 32 := by
sorry

end regular_polygon_perimeter_l3786_378694


namespace will_baseball_cards_pages_l3786_378627

/-- The number of pages needed to organize baseball cards in a binder -/
def pages_needed (cards_per_page new_cards old_cards : ℕ) : ℕ :=
  (new_cards + old_cards) / cards_per_page

/-- Theorem: Will uses 6 pages to organize his baseball cards -/
theorem will_baseball_cards_pages : pages_needed 3 8 10 = 6 := by
  sorry

end will_baseball_cards_pages_l3786_378627


namespace hyperbola_asymptotes_l3786_378678

/-- The equation of asymptotes for a hyperbola with given parameters -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) (he : Real.sqrt ((a^2 + b^2) / a^2) = 2) :
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x) :=
sorry

end hyperbola_asymptotes_l3786_378678


namespace range_of_a_l3786_378651

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |x - 2|

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f x a ≤ |x - 4|) ↔ a ∈ Set.Icc (-3) 0 :=
sorry

end range_of_a_l3786_378651


namespace tan_2theta_minus_pi_6_l3786_378607

theorem tan_2theta_minus_pi_6 (θ : ℝ) 
  (h : 4 * Real.cos (θ + π/3) * Real.cos (θ - π/6) = Real.sin (2*θ)) : 
  Real.tan (2*θ - π/6) = Real.sqrt 3 / 9 := by
  sorry

end tan_2theta_minus_pi_6_l3786_378607


namespace root_count_theorem_l3786_378696

def count_roots (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem root_count_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 - x) = f (2 + x))
  (h2 : ∀ x, f (7 - x) = f (7 + x))
  (h3 : ∀ x ∈ Set.Icc 0 7, f x = 0 ↔ x = 1 ∨ x = 3) :
  count_roots f (-2005) 2005 = 802 :=
sorry

end root_count_theorem_l3786_378696


namespace trig_identity_proof_l3786_378642

theorem trig_identity_proof : 
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) - 
  Real.cos (21 * π / 180) * Real.sin (81 * π / 180) = 
  -Real.sqrt 3 / 2 := by sorry

end trig_identity_proof_l3786_378642


namespace complex_number_quadrant_l3786_378671

theorem complex_number_quadrant : ∃ (z : ℂ), z = 2 * Complex.I^3 / (1 - Complex.I) ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end complex_number_quadrant_l3786_378671


namespace arithmetic_sequence_sum_l3786_378626

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 3 + a 11 = 3 →
  a 5 + a 6 + a 10 = 9/2 := by
  sorry

end arithmetic_sequence_sum_l3786_378626


namespace union_equals_A_l3786_378621

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 24 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2*a)*(x - a) < 0}

-- State the theorem
theorem union_equals_A (a : ℝ) : A ∪ B a = A ↔ a ∈ Set.Icc (-3/2) 4 := by
  sorry

end union_equals_A_l3786_378621


namespace saltwater_animals_per_aquarium_l3786_378605

theorem saltwater_animals_per_aquarium 
  (num_aquariums : ℕ) 
  (total_animals : ℕ) 
  (h1 : num_aquariums = 26) 
  (h2 : total_animals = 52) 
  (h3 : total_animals % num_aquariums = 0) :
  total_animals / num_aquariums = 2 := by
sorry

end saltwater_animals_per_aquarium_l3786_378605


namespace exactly_four_pairs_l3786_378687

/-- A line is tangent to a circle if and only if the distance from the center
    of the circle to the line equals the radius of the circle. -/
def is_tangent_line (m : ℕ) (n : ℕ) : Prop :=
  2^m = 2*n

/-- The condition that n and m are positive integers with n - m < 5 -/
def satisfies_condition (m : ℕ) (n : ℕ) : Prop :=
  0 < m ∧ 0 < n ∧ n < m + 5

/-- The main theorem stating that there are exactly 4 pairs (m, n) satisfying
    both the tangency condition and the inequality condition -/
theorem exactly_four_pairs :
  ∃! (s : Finset (ℕ × ℕ)),
    s.card = 4 ∧
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      (is_tangent_line p.1 p.2 ∧ satisfies_condition p.1 p.2)) := by
  sorry

end exactly_four_pairs_l3786_378687


namespace value_of_a_l3786_378661

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 5) 
  (eq3 : c = 3) : 
  a = 1 := by
sorry

end value_of_a_l3786_378661


namespace unique_multiplication_property_l3786_378613

theorem unique_multiplication_property : ∃! n : ℕ, 
  (n ≥ 10000000 ∧ n < 100000000) ∧  -- 8-digit number
  (n % 10 = 9) ∧                    -- ends in 9
  (∃ k : ℕ, n * 9 = k * 111111111)  -- when multiplied by 9, equals k * 111111111
    := by sorry

end unique_multiplication_property_l3786_378613


namespace n_times_n_plus_one_div_by_three_l3786_378634

theorem n_times_n_plus_one_div_by_three (n : ℤ) (h : 1 ≤ n ∧ n ≤ 99) :
  ∃ k : ℤ, n * (n + 1) = 3 * k := by
  sorry

end n_times_n_plus_one_div_by_three_l3786_378634


namespace function_equation_solution_l3786_378629

theorem function_equation_solution (a b : ℚ) :
  ∀ f : ℚ → ℚ, (∀ x y : ℚ, f (x + a + f y) = f (x + b) + y) →
  (∀ x : ℚ, f x = x + b - a) ∨ (∀ x : ℚ, f x = -x + b - a) :=
by sorry

end function_equation_solution_l3786_378629


namespace committees_with_president_count_l3786_378631

/-- The number of different five-student committees with an elected president
    that can be chosen from a group of ten students -/
def committees_with_president : ℕ :=
  (Nat.choose 10 5) * 5

/-- Theorem stating that the number of committees with a president is 1260 -/
theorem committees_with_president_count :
  committees_with_president = 1260 := by
  sorry

end committees_with_president_count_l3786_378631


namespace odd_even_function_inequalities_l3786_378602

-- Define the properties of functions f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y
def coincide_nonneg (f g : ℝ → ℝ) : Prop := ∀ x, x ≥ 0 → f x = g x

-- State the theorem
theorem odd_even_function_inequalities
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h_incr : is_increasing f)
  (h_coinc : coincide_nonneg f g)
  {a b : ℝ}
  (h_ab : a > b)
  (h_b_pos : b > 0) :
  (f b - f (-a) > g a - g (-b)) ∧
  (f a - f (-b) > g b - g (-a)) :=
by sorry

end odd_even_function_inequalities_l3786_378602


namespace solve_fraction_equation_l3786_378689

theorem solve_fraction_equation :
  ∀ x : ℚ, (2 / 3 : ℚ) - (1 / 4 : ℚ) = 1 / x → x = 12 / 5 := by
  sorry

end solve_fraction_equation_l3786_378689


namespace gymnasts_count_l3786_378643

/-- The number of gymnastics teams --/
def num_teams : ℕ := 4

/-- The total number of handshakes --/
def total_handshakes : ℕ := 595

/-- The number of gymnasts each coach shakes hands with --/
def coach_handshakes : ℕ := 6

/-- The total number of gymnasts across all teams --/
def total_gymnasts : ℕ := 34

/-- Theorem stating that the total number of gymnasts is 34 --/
theorem gymnasts_count : 
  (total_gymnasts * (total_gymnasts - 1)) / 2 + num_teams * coach_handshakes = total_handshakes :=
by sorry

end gymnasts_count_l3786_378643


namespace equation_has_real_root_l3786_378677

/-- The equation x^4 + 2px^3 + x^3 + 2px + 1 = 0 has at least one real root for all real p. -/
theorem equation_has_real_root (p : ℝ) : ∃ x : ℝ, x^4 + 2*p*x^3 + x^3 + 2*p*x + 1 = 0 := by
  sorry

end equation_has_real_root_l3786_378677


namespace bat_wings_area_l3786_378684

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Calculates the area of a triangle given three points -/
def triangleArea (a b c : Point) : ℝ :=
  0.5 * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

/-- Theorem: The area of the "bat wings" in the given rectangle configuration is 7.5 -/
theorem bat_wings_area (rect : Rectangle)
  (h_width : rect.bottomRight.x - rect.topLeft.x = 4)
  (h_height : rect.bottomRight.y - rect.topLeft.y = 5)
  (j : Point) (k : Point) (l : Point) (m : Point)
  (h_j : j = rect.topLeft)
  (h_k : k.x - j.x = 2 ∧ k.y = rect.bottomRight.y)
  (h_l : l.x = rect.bottomRight.x ∧ l.y - k.y = 2)
  (h_m : m.x = rect.topLeft.x ∧ m.y = rect.bottomRight.y)
  (h_mj : m.y - j.y = 2)
  (h_jk : k.x - j.x = 2)
  (h_kl : l.x - k.x = 2) :
  triangleArea j m k + triangleArea j k l = 7.5 := by
  sorry

end bat_wings_area_l3786_378684


namespace binomial_18_4_l3786_378664

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end binomial_18_4_l3786_378664


namespace hexagonal_pyramid_impossible_equal_edge_and_slant_l3786_378672

structure RegularPyramid (n : ℕ) where
  baseEdgeLength : ℝ
  slantHeight : ℝ

/-- Theorem: In a regular hexagonal pyramid, it's impossible for the base edge length
    to be equal to the slant height. -/
theorem hexagonal_pyramid_impossible_equal_edge_and_slant :
  ¬∃ (p : RegularPyramid 6), p.baseEdgeLength = p.slantHeight :=
sorry

end hexagonal_pyramid_impossible_equal_edge_and_slant_l3786_378672


namespace square_diff_fourth_power_l3786_378695

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end square_diff_fourth_power_l3786_378695


namespace cashier_payment_problem_l3786_378638

theorem cashier_payment_problem :
  (¬ ∃ x y : ℤ, 72 * x + 105 * y = 1) ∧
  (∃ x y : ℤ, 72 * x + 105 * y = 3) := by
  sorry

end cashier_payment_problem_l3786_378638


namespace greatest_piece_length_l3786_378665

theorem greatest_piece_length (rope1 rope2 rope3 max_length : ℕ) 
  (h1 : rope1 = 48)
  (h2 : rope2 = 72)
  (h3 : rope3 = 120)
  (h4 : max_length = 24) : 
  (Nat.gcd rope1 (Nat.gcd rope2 rope3) ≤ max_length ∧ 
   Nat.gcd rope1 (Nat.gcd rope2 rope3) = max_length) := by
  sorry

#eval Nat.gcd 48 (Nat.gcd 72 120)

end greatest_piece_length_l3786_378665


namespace not_exist_prime_power_of_six_plus_nineteen_l3786_378603

theorem not_exist_prime_power_of_six_plus_nineteen :
  ∀ n : ℕ, ¬ Nat.Prime (6^n + 19) := by
  sorry

end not_exist_prime_power_of_six_plus_nineteen_l3786_378603


namespace fruit_store_discount_l3786_378633

/-- Represents the discount policy of a fruit store -/
def discount_policy (lemon_price papaya_price mango_price : ℕ)
                    (lemon_qty papaya_qty mango_qty : ℕ)
                    (total_paid : ℕ) : ℕ :=
  let total_cost := lemon_price * lemon_qty + papaya_price * papaya_qty + mango_price * mango_qty
  let total_fruits := lemon_qty + papaya_qty + mango_qty
  total_cost - total_paid

theorem fruit_store_discount :
  discount_policy 2 1 4 6 4 2 21 = 3 :=
by
  sorry

end fruit_store_discount_l3786_378633


namespace triangle_area_proof_l3786_378685

/-- The slope of the first line -/
def m₁ : ℚ := 3/4

/-- The slope of the second line -/
def m₂ : ℚ := -3/2

/-- The x-coordinate of the intersection point of the first two lines -/
def x₀ : ℚ := 4

/-- The y-coordinate of the intersection point of the first two lines -/
def y₀ : ℚ := 3

/-- The equation of the third line: ax + by = c -/
def a : ℚ := 1
def b : ℚ := 2
def c : ℚ := 12

/-- The area of the triangle -/
def triangle_area : ℚ := 15/4

theorem triangle_area_proof :
  let line1 := fun x => m₁ * (x - x₀) + y₀
  let line2 := fun x => m₂ * (x - x₀) + y₀
  let line3 := fun x y => a * x + b * y = c
  ∃ x₁ y₁ x₂ y₂,
    line1 x₁ = y₁ ∧ line3 x₁ y₁ ∧
    line2 x₂ = y₂ ∧ line3 x₂ y₂ ∧
    triangle_area = (1/2) * abs ((x₀ * (y₁ - y₂) + x₁ * (y₂ - y₀) + x₂ * (y₀ - y₁))) :=
sorry

end triangle_area_proof_l3786_378685


namespace shoe_selection_outcomes_l3786_378615

/-- The number of distinct pairs of shoes -/
def num_pairs : ℕ := 10

/-- The number of shoes drawn -/
def num_drawn : ℕ := 4

/-- The number of ways to select 4 shoes such that none form a pair -/
def no_pairs : ℕ := (Nat.choose num_pairs num_drawn) * (2^num_drawn)

/-- The number of ways to select 4 shoes such that two form a pair and the other two do not form pairs -/
def one_pair : ℕ := (Nat.choose num_pairs 2) * (2^2) * (Nat.choose (num_pairs - 2) 1)

/-- The number of ways to select 4 shoes such that they form two complete pairs -/
def two_pairs : ℕ := Nat.choose num_pairs 2

theorem shoe_selection_outcomes :
  no_pairs = 3360 ∧ one_pair = 1440 ∧ two_pairs = 45 := by
  sorry

end shoe_selection_outcomes_l3786_378615


namespace digit_difference_of_63_l3786_378632

theorem digit_difference_of_63 :
  let tens : ℕ := 63 / 10
  let ones : ℕ := 63 % 10
  tens + ones = 9 →
  tens - ones = 3 :=
by sorry

end digit_difference_of_63_l3786_378632


namespace two_thousand_one_in_first_column_l3786_378649

-- Define the column patterns
def first_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 1
def second_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 3
def third_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 5
def fourth_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 7

-- Define the theorem
theorem two_thousand_one_in_first_column : 
  first_column 2001 ∧ ¬(second_column 2001 ∨ third_column 2001 ∨ fourth_column 2001) :=
by sorry

end two_thousand_one_in_first_column_l3786_378649


namespace quadratic_equation_solution_l3786_378635

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 + 4*x - 1
  ∃ x₁ x₂ : ℝ, (x₁ = -2 + Real.sqrt 5 ∧ x₂ = -2 - Real.sqrt 5) ∧ 
              (f x₁ = 0 ∧ f x₂ = 0) ∧
              (∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_equation_solution_l3786_378635


namespace power_sum_difference_l3786_378628

theorem power_sum_difference : 2^(1+2+3) - (2^1 + 2^2 + 2^3) = 50 := by
  sorry

end power_sum_difference_l3786_378628


namespace no_solution_exists_l3786_378676

theorem no_solution_exists : ¬∃ x : ℝ, (81 : ℝ)^(3*x) = (27 : ℝ)^(4*x - 5) := by
  sorry

end no_solution_exists_l3786_378676


namespace cube_volume_from_surface_area_l3786_378622

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 125 := by
  sorry

end cube_volume_from_surface_area_l3786_378622


namespace triangle_area_implies_k_difference_l3786_378640

-- Define the lines
def line1 (k₁ b x : ℝ) : ℝ := k₁ * x + 3 * k₁ + b
def line2 (k₂ b x : ℝ) : ℝ := k₂ * x + 3 * k₂ + b

-- Define the theorem
theorem triangle_area_implies_k_difference
  (k₁ k₂ b : ℝ)
  (h1 : k₁ * k₂ < 0)
  (h2 : (1/2) * 3 * |3 * k₁ - 3 * k₂| * 3 = 9) :
  |k₁ - k₂| = 2 := by
sorry

end triangle_area_implies_k_difference_l3786_378640


namespace road_repair_theorem_l3786_378630

/-- Represents the road repair scenario -/
structure RoadRepair where
  initial_workers : ℕ
  initial_days : ℕ
  worked_days : ℕ
  additional_workers : ℕ

/-- Calculates the number of additional days needed to complete the work -/
def additional_days_needed (repair : RoadRepair) : ℚ :=
  let total_work := repair.initial_workers * repair.initial_days
  let work_done := repair.initial_workers * repair.worked_days
  let remaining_work := total_work - work_done
  let new_workforce := repair.initial_workers + repair.additional_workers
  remaining_work / new_workforce

/-- The theorem stating that 6 additional days are needed to complete the work -/
theorem road_repair_theorem (repair : RoadRepair)
  (h1 : repair.initial_workers = 24)
  (h2 : repair.initial_days = 12)
  (h3 : repair.worked_days = 4)
  (h4 : repair.additional_workers = 8) :
  additional_days_needed repair = 6 := by
  sorry

#eval additional_days_needed ⟨24, 12, 4, 8⟩

end road_repair_theorem_l3786_378630


namespace phoenix_temperature_l3786_378692

theorem phoenix_temperature (t : ℝ) : 
  (∀ s, -s^2 + 14*s + 40 = 77 → s ≤ t) → -t^2 + 14*t + 40 = 77 → t = 11 := by
  sorry

end phoenix_temperature_l3786_378692


namespace decreasing_linear_function_conditions_l3786_378611

/-- A linear function y = kx - b where y decreases as x increases
    and intersects the y-axis above the x-axis -/
def DecreasingLinearFunction (k b : ℝ) : Prop :=
  k < 0 ∧ b > 0

/-- Theorem stating that for a linear function y = kx - b,
    if y decreases as x increases and intersects the y-axis above the x-axis,
    then k < 0 and b > 0 -/
theorem decreasing_linear_function_conditions (k b : ℝ) :
  DecreasingLinearFunction k b ↔ k < 0 ∧ b > 0 := by
  sorry

end decreasing_linear_function_conditions_l3786_378611


namespace nested_expression_simplification_l3786_378688

theorem nested_expression_simplification (x : ℝ) : 1 - (1 - (1 - (1 + (1 - (1 - x))))) = 2 - x := by
  sorry

end nested_expression_simplification_l3786_378688


namespace cows_not_black_l3786_378618

theorem cows_not_black (total : ℕ) (black : ℕ) : total = 18 → black = (total / 2 + 5) → total - black = 4 := by
  sorry

end cows_not_black_l3786_378618


namespace nate_age_when_ember_is_14_l3786_378619

-- Define the initial ages
def nate_initial_age : ℕ := 14
def ember_initial_age : ℕ := nate_initial_age / 2

-- Define the target age for Ember
def ember_target_age : ℕ := 14

-- Calculate the age difference
def age_difference : ℕ := ember_target_age - ember_initial_age

-- Theorem statement
theorem nate_age_when_ember_is_14 :
  nate_initial_age + age_difference = 21 :=
sorry

end nate_age_when_ember_is_14_l3786_378619


namespace family_gathering_problem_l3786_378657

theorem family_gathering_problem (P : ℕ) : 
  P / 2 = P - 10 → 
  P / 2 + P / 4 + (P - (P / 2 + P / 4)) = P → 
  P = 20 := by
  sorry

end family_gathering_problem_l3786_378657


namespace nearest_integer_to_power_l3786_378691

theorem nearest_integer_to_power : 
  ∃ n : ℤ, |n - (3 + Real.sqrt 5)^6| < 1/2 ∧ n = 2744 :=
sorry

end nearest_integer_to_power_l3786_378691


namespace cosine_range_in_triangle_l3786_378681

theorem cosine_range_in_triangle (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_AB : dist A B = 3)
  (h_AC : dist A C = 2)
  (h_BC : dist B C > Real.sqrt 2) :
  ∃ (cosA : ℝ), cosA = (dist A B)^2 + (dist A C)^2 - (dist B C)^2 / (2 * dist A B * dist A C) ∧ 
  -1 < cosA ∧ cosA < 11/12 :=
sorry

end cosine_range_in_triangle_l3786_378681


namespace equation_solution_l3786_378624

theorem equation_solution : ∃ x : ℚ, x ≠ 0 ∧ (3 / x - (3 / x) / (9 / x) = 1 / 2) ∧ x = 6 / 5 := by
  sorry

end equation_solution_l3786_378624


namespace trig_identity_l3786_378674

theorem trig_identity (α : ℝ) : 
  1 - Real.cos (2 * α - π) + Real.cos (4 * α - 2 * π) = 
  4 * Real.cos (2 * α) * Real.cos (π / 6 + α) * Real.cos (π / 6 - α) := by
  sorry

end trig_identity_l3786_378674


namespace pen_retailer_profit_percentage_specific_pen_retailer_profit_l3786_378679

/-- Calculates the profit percentage for a retailer selling pens with a discount -/
theorem pen_retailer_profit_percentage 
  (num_pens : ℕ) 
  (price_per_36_pens : ℝ) 
  (discount_percent : ℝ) : ℝ :=
let cost_per_pen := price_per_36_pens / 36
let total_cost := num_pens * cost_per_pen
let selling_price_per_pen := price_per_36_pens / 36 * (1 - discount_percent / 100)
let total_selling_price := num_pens * selling_price_per_pen
let profit := total_selling_price - total_cost
let profit_percentage := (profit / total_cost) * 100
profit_percentage

/-- The profit percentage for a retailer buying 120 pens at the price of 36 pens 
    and selling with a 1% discount is 230% -/
theorem specific_pen_retailer_profit :
  pen_retailer_profit_percentage 120 36 1 = 230 := by
  sorry

end pen_retailer_profit_percentage_specific_pen_retailer_profit_l3786_378679


namespace equal_milk_water_ratio_l3786_378699

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The first mixture with milk:water ratio 5:4 -/
def mixture_p : Mixture := ⟨5, 4⟩

/-- The second mixture with milk:water ratio 2:7 -/
def mixture_q : Mixture := ⟨2, 7⟩

/-- Combines two mixtures in a given ratio -/
def combine_mixtures (m1 m2 : Mixture) (r1 r2 : ℚ) : Mixture :=
  ⟨r1 * m1.milk + r2 * m2.milk, r1 * m1.water + r2 * m2.water⟩

/-- Theorem stating that combining mixture_p and mixture_q in ratio 5:1 results in equal milk and water -/
theorem equal_milk_water_ratio :
  let result := combine_mixtures mixture_p mixture_q 5 1
  result.milk = result.water := by sorry

end equal_milk_water_ratio_l3786_378699


namespace sqrt_factorial_five_squared_l3786_378601

theorem sqrt_factorial_five_squared (n : ℕ) : n = 5 → Real.sqrt ((n.factorial : ℝ) * n.factorial) = 120 := by
  sorry

end sqrt_factorial_five_squared_l3786_378601


namespace basketball_team_score_l3786_378614

theorem basketball_team_score :
  ∀ (chandra akiko michiko bailey damien ella : ℕ),
    chandra = 2 * akiko →
    akiko = michiko + 4 →
    michiko * 2 = bailey →
    bailey = 14 →
    damien = 3 * akiko →
    ella = chandra + (chandra / 5) →
    chandra + akiko + michiko + bailey + damien + ella = 113 := by
  sorry

end basketball_team_score_l3786_378614


namespace vector_dot_product_l3786_378620

/-- Given two vectors a and b in ℝ², prove that their dot product is -12
    when their sum is (1, -3) and their difference is (3, 7). -/
theorem vector_dot_product (a b : ℝ × ℝ) 
    (h1 : a.1 + b.1 = 1 ∧ a.2 + b.2 = -3)
    (h2 : a.1 - b.1 = 3 ∧ a.2 - b.2 = 7) :
    a.1 * b.1 + a.2 * b.2 = -12 := by
  sorry

end vector_dot_product_l3786_378620


namespace cylinder_lateral_surface_area_l3786_378669

/-- The lateral surface area of a cylinder with base diameter and height both equal to 2 cm is 4π cm² -/
theorem cylinder_lateral_surface_area :
  ∀ (d h r : ℝ),
  d = 2 →  -- base diameter is 2 cm
  h = 2 →  -- height is 2 cm
  r = d / 2 →  -- radius is half the diameter
  2 * Real.pi * r * h = 4 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l3786_378669


namespace max_ab_value_l3786_378693

noncomputable def g (x : ℝ) : ℝ := 2^x

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : g a * g b = 2) :
  ∀ (x y : ℝ), x > 0 → y > 0 → g x * g y = 2 → x * y ≤ a * b ∧ a * b = 1/4 :=
by sorry

end max_ab_value_l3786_378693


namespace action_figures_earnings_l3786_378637

/-- Calculates the total earnings from selling action figures with discounts -/
def total_earnings (type_a_count type_b_count type_c_count type_d_count : ℕ)
                   (type_a_value type_b_value type_c_value type_d_value : ℕ)
                   (type_a_discount type_b_discount type_c_discount type_d_discount : ℕ) : ℕ :=
  (type_a_count * (type_a_value - type_a_discount)) +
  (type_b_count * (type_b_value - type_b_discount)) +
  (type_c_count * (type_c_value - type_c_discount)) +
  (type_d_count * (type_d_value - type_d_discount))

/-- Theorem stating that the total earnings from selling all action figures is $435 -/
theorem action_figures_earnings :
  total_earnings 6 5 4 5 22 35 45 50 10 14 18 20 = 435 := by
  sorry

end action_figures_earnings_l3786_378637


namespace inequality_proof_l3786_378698

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (4 / (c - a)) ≥ 0 := by
  sorry

end inequality_proof_l3786_378698


namespace candy_bar_difference_l3786_378673

/-- Given information about candy bars possessed by Lena, Kevin, and Nicole, 
    prove that Lena has 19.6 more candy bars than Nicole. -/
theorem candy_bar_difference (lena kevin nicole : ℝ) : 
  lena = 37.5 ∧ 
  lena + 9.5 = 5 * kevin ∧ 
  kevin = nicole - 8.5 → 
  lena - nicole = 19.6 := by
  sorry

end candy_bar_difference_l3786_378673


namespace bird_ratio_l3786_378639

/-- Proves that the ratio of cardinals to bluebirds is 3:1 given the conditions of the bird problem -/
theorem bird_ratio (cardinals bluebirds swallows : ℕ) 
  (swallow_half : swallows = bluebirds / 2)
  (swallow_count : swallows = 2)
  (total_birds : cardinals + bluebirds + swallows = 18) :
  cardinals = 3 * bluebirds := by
  sorry


end bird_ratio_l3786_378639


namespace remaining_chess_pieces_l3786_378653

def standard_chess_pieces : ℕ := 32
def initial_player_pieces : ℕ := 16
def arianna_lost_pieces : ℕ := 3
def samantha_lost_pieces : ℕ := 9

theorem remaining_chess_pieces :
  standard_chess_pieces - (arianna_lost_pieces + samantha_lost_pieces) = 20 :=
by sorry

end remaining_chess_pieces_l3786_378653


namespace julie_work_hours_julie_school_year_hours_l3786_378616

/-- Given Julie's work schedule and earnings, calculate her required weekly hours during the school year --/
theorem julie_work_hours (summer_weekly_hours : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) : ℕ :=
  let hourly_rate := summer_earnings / (summer_weekly_hours * summer_weeks)
  let school_year_hours := school_year_target / hourly_rate
  let school_year_weekly_hours := school_year_hours / school_year_weeks
  school_year_weekly_hours

/-- Prove that Julie needs to work 15 hours per week during the school year --/
theorem julie_school_year_hours : 
  julie_work_hours 60 10 8000 50 10000 = 15 := by
  sorry

end julie_work_hours_julie_school_year_hours_l3786_378616


namespace quadratic_function_m_range_l3786_378658

theorem quadratic_function_m_range
  (a : ℝ) (m : ℝ) (y₁ y₂ : ℝ)
  (h_a_neg : a < 0)
  (h_y₁ : y₁ = a * m^2 - 4 * a * m)
  (h_y₂ : y₂ = a * (2*m)^2 - 4 * a * (2*m))
  (h_above_line : y₁ > -3*a ∧ y₂ > -3*a)
  (h_y₁_gt_y₂ : y₁ > y₂) :
  4/3 < m ∧ m < 3/2 := by
sorry

end quadratic_function_m_range_l3786_378658


namespace expand_expression_l3786_378646

theorem expand_expression (y : ℝ) : 12 * (3 * y + 7) = 36 * y + 84 := by
  sorry

end expand_expression_l3786_378646


namespace modified_cube_surface_area_l3786_378675

/-- Represents a 9x9x9 cube composed of 3x3x3 subcubes -/
structure LargeCube where
  subcubes : Fin 3 → Fin 3 → Fin 3 → Unit

/-- Represents the modified structure after removing center cubes and facial units -/
structure ModifiedCube where
  remaining_subcubes : Fin 20 → Unit
  removed_centers : Unit
  removed_facial_units : Unit

/-- Calculates the surface area of the modified cube structure -/
def surface_area (cube : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that the surface area of the modified cube is 1056 square units -/
theorem modified_cube_surface_area :
  ∀ (cube : LargeCube),
  ∃ (modified : ModifiedCube),
  surface_area modified = 1056 :=
sorry

end modified_cube_surface_area_l3786_378675


namespace problem_statement_l3786_378604

theorem problem_statement (x y : ℝ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : Real.log x / Real.log 4 ^ 3 + Real.log y / Real.log 5 ^ 3 + 9 = 
        12 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) : 
  x^2 + y^2 = 64 + 25 * Real.sqrt 5 := by
  sorry

end problem_statement_l3786_378604


namespace josh_candy_count_l3786_378625

def candy_problem (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) (shared_candies : ℕ) : ℕ :=
  let remaining_after_siblings := initial_candies - siblings * candies_per_sibling
  let remaining_after_friend := remaining_after_siblings / 2
  remaining_after_friend - shared_candies

theorem josh_candy_count : candy_problem 100 3 10 19 = 16 := by
  sorry

end josh_candy_count_l3786_378625


namespace factorial_15_l3786_378644

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_15 : factorial 15 = 1307674368000 := by sorry

end factorial_15_l3786_378644


namespace transmission_time_is_6_4_minutes_l3786_378641

/-- Represents the number of blocks to be sent -/
def num_blocks : ℕ := 80

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 768

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 160

/-- Represents the total number of chunks to be sent -/
def total_chunks : ℕ := num_blocks * chunks_per_block

/-- Represents the time in seconds to send all chunks -/
def transmission_time_seconds : ℚ := total_chunks / transmission_rate

/-- Represents the time in minutes to send all chunks -/
def transmission_time_minutes : ℚ := transmission_time_seconds / 60

/-- Theorem stating that the transmission time is 6.4 minutes -/
theorem transmission_time_is_6_4_minutes : transmission_time_minutes = 6.4 := by
  sorry

end transmission_time_is_6_4_minutes_l3786_378641


namespace english_not_russian_count_l3786_378647

/-- Represents the set of teachers who know English -/
def E : Finset Nat := sorry

/-- Represents the set of teachers who know Russian -/
def R : Finset Nat := sorry

theorem english_not_russian_count :
  (E.card = 75) →
  (R.card = 55) →
  ((E ∩ R).card = 110) →
  ((E \ R).card = 55) := by
  sorry

end english_not_russian_count_l3786_378647


namespace julia_age_l3786_378668

-- Define the ages of the individuals
def Grace : ℕ := 20
def Helen : ℕ := Grace + 4
def Ian : ℕ := Helen - 5
def Julia : ℕ := Ian + 2

-- Theorem to prove
theorem julia_age : Julia = 21 := by
  sorry

end julia_age_l3786_378668


namespace rectangle_area_l3786_378652

theorem rectangle_area (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l3786_378652


namespace bella_prob_reach_edge_l3786_378670

/-- Represents a position on the 4x4 grid -/
inductive Position
| Central : Position
| NearEdge : Position
| Edge : Position

/-- Represents the possible directions of movement -/
inductive Direction
| Up | Down | Left | Right

/-- Represents the state of Bella's movement -/
structure BellaState where
  position : Position
  hops : Nat

/-- Transition function for Bella's movement -/
def transition (state : BellaState) (dir : Direction) : BellaState :=
  match state.position with
  | Position.Central => ⟨Position.NearEdge, state.hops + 1⟩
  | Position.NearEdge => 
      if state.hops < 5 then ⟨Position.Edge, state.hops + 1⟩
      else state
  | Position.Edge => state

/-- Probability of reaching an edge square within 5 hops -/
def prob_reach_edge (state : BellaState) : ℚ :=
  sorry

/-- Main theorem: Probability of reaching an edge square within 5 hops is 7/8 -/
theorem bella_prob_reach_edge :
  prob_reach_edge ⟨Position.Central, 0⟩ = 7/8 :=
sorry

end bella_prob_reach_edge_l3786_378670


namespace corn_plants_multiple_of_max_l3786_378650

/-- Represents the number of plants in a garden -/
structure GardenPlants where
  sunflowers : ℕ
  corn : ℕ
  tomatoes : ℕ

/-- Represents the constraints for planting in the garden -/
structure GardenConstraints where
  max_plants_per_row : ℕ
  same_plants_per_row : Bool
  one_type_per_row : Bool

/-- Theorem stating that the number of corn plants must be a multiple of the maximum plants per row -/
theorem corn_plants_multiple_of_max (garden : GardenPlants) (constraints : GardenConstraints) 
  (h1 : garden.sunflowers = 45)
  (h2 : garden.tomatoes = 63)
  (h3 : constraints.max_plants_per_row = 9)
  (h4 : constraints.same_plants_per_row = true)
  (h5 : constraints.one_type_per_row = true) :
  ∃ k : ℕ, garden.corn = k * constraints.max_plants_per_row := by
  sorry

end corn_plants_multiple_of_max_l3786_378650


namespace reciprocal_of_2023_l3786_378654

theorem reciprocal_of_2023 :
  ∀ (x : ℝ), x = 2023 → x⁻¹ = (1 : ℝ) / 2023 := by
  sorry

end reciprocal_of_2023_l3786_378654


namespace cubic_polynomial_with_coefficient_roots_l3786_378623

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The polynomial function for a CubicPolynomial -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℚ) : ℚ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Predicate for a CubicPolynomial having its coefficients as roots -/
def CubicPolynomial.hasCoefficientsAsRoots (p : CubicPolynomial) : Prop :=
  p.eval p.a = 0 ∧ p.eval p.b = 0 ∧ p.eval p.c = 0

/-- The two specific polynomials mentioned in the problem -/
def f₁ : CubicPolynomial := ⟨1, -2, 0⟩
def f₂ : CubicPolynomial := ⟨1, -1, -1⟩

/-- The main theorem stating that f₁ and f₂ are the only valid polynomials -/
theorem cubic_polynomial_with_coefficient_roots :
  ∀ p : CubicPolynomial, p.hasCoefficientsAsRoots → p = f₁ ∨ p = f₂ := by
  sorry

end cubic_polynomial_with_coefficient_roots_l3786_378623


namespace gcd_of_256_180_600_l3786_378600

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 4 := by
  sorry

end gcd_of_256_180_600_l3786_378600


namespace wendy_chocolate_sales_l3786_378659

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (unsold_bars : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Proves that Wendy made $18 from selling chocolate bars -/
theorem wendy_chocolate_sales : money_made 9 3 3 = 18 := by
  sorry

end wendy_chocolate_sales_l3786_378659


namespace quadratic_polynomial_root_l3786_378660

theorem quadratic_polynomial_root : ∃ (a b c : ℝ), 
  (a = 1) ∧ 
  (∀ x : ℂ, x^2 + b*x + c = 0 ↔ x = 4 + I ∨ x = 4 - I) ∧
  (a*x^2 + b*x + c = x^2 - 8*x + 17) :=
sorry

end quadratic_polynomial_root_l3786_378660


namespace smallest_solution_of_equation_l3786_378608

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 1 / (x - 3) + 1 / (x - 5) - 4 / (x - 4)
  ∃ (sol : ℝ), f sol = 0 ∧ sol = 4 - Real.sqrt 2 ∧ ∀ x, f x = 0 → x ≥ sol :=
by sorry

end smallest_solution_of_equation_l3786_378608


namespace correct_answers_for_zero_score_l3786_378606

theorem correct_answers_for_zero_score (total_questions : ℕ) 
  (points_per_correct : ℕ) (points_per_wrong : ℕ) : 
  total_questions = 26 →
  points_per_correct = 8 →
  points_per_wrong = 5 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    points_per_correct * correct_answers = 
      points_per_wrong * (total_questions - correct_answers) ∧
    correct_answers = 10 := by
  sorry

end correct_answers_for_zero_score_l3786_378606


namespace bicycle_distance_l3786_378656

theorem bicycle_distance (back_wheel_perimeter front_wheel_perimeter : ℝ)
  (revolution_difference : ℕ) (distance : ℝ) :
  back_wheel_perimeter = 9 →
  front_wheel_perimeter = 7 →
  revolution_difference = 10 →
  distance / front_wheel_perimeter = distance / back_wheel_perimeter + revolution_difference →
  distance = 315 := by
sorry

end bicycle_distance_l3786_378656


namespace constant_kill_time_l3786_378610

/-- Represents the time in minutes it takes for a given number of cats to kill the same number of rats -/
def killTime (n : ℕ) : ℝ :=
  3  -- We define this as 3 based on the given condition for 100 cats and 100 rats

theorem constant_kill_time (n : ℕ) (h : n ≥ 3) : killTime n = 3 := by
  sorry

#check constant_kill_time

end constant_kill_time_l3786_378610


namespace alia_markers_count_l3786_378683

theorem alia_markers_count : ∀ (steve austin alia : ℕ),
  steve = 60 →
  austin = steve / 3 →
  alia = 2 * austin →
  alia = 40 := by
sorry

end alia_markers_count_l3786_378683


namespace fifth_student_guess_l3786_378609

def jellybeanGuess (first second third fourth fifth : ℕ) : Prop :=
  second = 8 * first ∧
  third = second - 200 ∧
  fourth = ((first + second + third) / 3 + 25 : ℕ) ∧
  fifth = fourth + (fourth * 20 / 100 : ℕ)

theorem fifth_student_guess :
  ∀ first second third fourth fifth : ℕ,
    first = 100 →
    jellybeanGuess first second third fourth fifth →
    fifth = 630 := by
  sorry

end fifth_student_guess_l3786_378609


namespace gcf_and_sum_proof_l3786_378648

def a : ℕ := 198
def b : ℕ := 396

theorem gcf_and_sum_proof : 
  (Nat.gcd a b = a) ∧ 
  (a + 4 * a = 990) := by
sorry

end gcf_and_sum_proof_l3786_378648


namespace connors_garage_wheels_l3786_378697

/-- Calculates the total number of wheels in Connor's garage -/
theorem connors_garage_wheels :
  let num_bicycles : ℕ := 20
  let num_cars : ℕ := 10
  let num_motorcycles : ℕ := 5
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_car : ℕ := 4
  let wheels_per_motorcycle : ℕ := 2
  (num_bicycles * wheels_per_bicycle + 
   num_cars * wheels_per_car + 
   num_motorcycles * wheels_per_motorcycle) = 90 := by
  sorry

end connors_garage_wheels_l3786_378697


namespace triangle_properties_l3786_378655

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition from the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  (t.b + 2 * t.a) * Real.cos t.C + t.c * Real.cos t.B = 0

/-- The angle bisector of C has length 2 -/
def angleBisectorLength (t : Triangle) : Prop :=
  2 = 2 * t.a * t.b * Real.sin (t.C / 2) / (t.a + t.b)

theorem triangle_properties (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : angleBisectorLength t) : 
  t.C = 2 * Real.pi / 3 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 2 * a + b ≥ 6 + 4 * Real.sqrt 2 := by
  sorry

end triangle_properties_l3786_378655
