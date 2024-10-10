import Mathlib

namespace replacement_results_in_four_terms_l2637_263700

-- Define the expression as a function of x and the replacement term
def expression (x : ℝ) (replacement : ℝ → ℝ) : ℝ := 
  (x^3 - 2)^2 + (x^2 + replacement x)^2

-- Define the expansion of the expression
def expanded_expression (x : ℝ) : ℝ := 
  x^6 + x^4 + 4*x^2 + 4

-- Theorem statement
theorem replacement_results_in_four_terms :
  ∀ x : ℝ, expression x (λ y => 2*y) = expanded_expression x :=
by sorry

end replacement_results_in_four_terms_l2637_263700


namespace complex_square_l2637_263727

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 5 - 3*i) (h2 : i^2 = -1) :
  z^2 = 16 - 30*i := by sorry

end complex_square_l2637_263727


namespace inscribed_triangle_radius_l2637_263712

/-- Given a regular triangle inscribed in a circular segment with the following properties:
    - The arc of the segment has a central angle α
    - One vertex of the triangle coincides with the midpoint of the arc
    - The other two vertices lie on the chord
    - The area of the triangle is S
    Then the radius R of the circle is given by R = (√(S√3)) / (2 sin²(α/4)) -/
theorem inscribed_triangle_radius (S α : ℝ) (h_S : S > 0) (h_α : 0 < α ∧ α < 2 * Real.pi) :
  ∃ R : ℝ, R > 0 ∧ R = (Real.sqrt (S * Real.sqrt 3)) / (2 * (Real.sin (α / 4))^2) :=
sorry

end inscribed_triangle_radius_l2637_263712


namespace sphere_identical_views_other_bodies_different_views_l2637_263752

-- Define the geometric bodies
inductive GeometricBody
  | Cylinder
  | Cone
  | Sphere
  | TriangularPyramid

-- Define a function to check if a geometric body has identical views
def hasIdenticalViews (body : GeometricBody) : Prop :=
  match body with
  | GeometricBody.Sphere => true
  | _ => false

-- Theorem stating that only a sphere has identical views
theorem sphere_identical_views :
  ∀ (body : GeometricBody),
    hasIdenticalViews body ↔ body = GeometricBody.Sphere :=
by sorry

-- Prove that other geometric bodies do not have identical views
theorem other_bodies_different_views :
  ¬(hasIdenticalViews GeometricBody.Cylinder) ∧
  ¬(hasIdenticalViews GeometricBody.Cone) ∧
  ¬(hasIdenticalViews GeometricBody.TriangularPyramid) :=
by sorry

end sphere_identical_views_other_bodies_different_views_l2637_263752


namespace guppies_per_day_l2637_263725

/-- The number of guppies Jason's moray eel eats per day -/
def moray_eel_guppies : ℕ := 20

/-- The number of betta fish Jason has -/
def num_betta_fish : ℕ := 5

/-- The number of guppies each betta fish eats per day -/
def betta_fish_guppies : ℕ := 7

/-- Theorem: Jason needs to buy 55 guppies per day -/
theorem guppies_per_day : 
  moray_eel_guppies + num_betta_fish * betta_fish_guppies = 55 := by
  sorry

end guppies_per_day_l2637_263725


namespace candy_problem_l2637_263788

theorem candy_problem :
  ∀ (S : ℕ) (N : ℕ),
    (∀ (i : ℕ), i < N → S / N = (S - S / N - 11)) →
    (S / N > 1) →
    (N > 1) →
    S = 33 :=
by sorry

end candy_problem_l2637_263788


namespace geometric_sequence_problem_l2637_263790

theorem geometric_sequence_problem (a : ℕ → ℚ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 5 = (5 : ℚ) / 3 →                   -- 5th term equals constant term of expansion
  a 3 * a 7 = (25 : ℚ) / 9 :=
by sorry

end geometric_sequence_problem_l2637_263790


namespace literary_readers_count_l2637_263729

theorem literary_readers_count (total : ℕ) (sci_fi : ℕ) (both : ℕ) (literary : ℕ) : 
  total = 150 → sci_fi = 120 → both = 60 → literary = total - sci_fi + both → literary = 90 :=
by
  sorry

end literary_readers_count_l2637_263729


namespace frog_eggs_eaten_percentage_l2637_263797

theorem frog_eggs_eaten_percentage
  (total_eggs : ℕ)
  (dry_up_percentage : ℚ)
  (hatch_fraction : ℚ)
  (hatched_frogs : ℕ)
  (h_total_eggs : total_eggs = 800)
  (h_dry_up_percentage : dry_up_percentage = 1 / 10)
  (h_hatch_fraction : hatch_fraction = 1 / 4)
  (h_hatched_frogs : hatched_frogs = 40) :
  (total_eggs : ℚ) * (1 - dry_up_percentage - hatch_fraction * (1 - dry_up_percentage)) = 70 / 100 * total_eggs :=
sorry

end frog_eggs_eaten_percentage_l2637_263797


namespace perpendicular_line_with_same_intercept_l2637_263744

/-- Given a line l with equation x/3 - y/4 = 1, 
    prove that the line with equation 3x + 4y + 16 = 0 
    has the same y-intercept as l and is perpendicular to l -/
theorem perpendicular_line_with_same_intercept 
  (x y : ℝ) (l : x / 3 - y / 4 = 1) :
  ∃ (m b : ℝ), 
    (-- Same y-intercept condition
     b = -4) ∧ 
    (-- Perpendicular condition
     m * (4 / 3) = -1) ∧
    (-- Equation of the new line
     3 * x + 4 * y + 16 = 0) := by
  sorry

end perpendicular_line_with_same_intercept_l2637_263744


namespace initial_peaches_calculation_l2637_263746

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked : ℕ := 42

/-- The total number of peaches Sally has now -/
def total_peaches_now : ℕ := 55

/-- The initial number of peaches at Sally's roadside fruit dish -/
def initial_peaches : ℕ := total_peaches_now - peaches_picked

theorem initial_peaches_calculation :
  initial_peaches = total_peaches_now - peaches_picked :=
by sorry

end initial_peaches_calculation_l2637_263746


namespace common_chord_of_circles_l2637_263770

/-- The equation of the common chord of two circles -/
theorem common_chord_of_circles (x y : ℝ) :
  (x^2 + y^2 - 4*x - 3 = 0) ∧ (x^2 + y^2 - 4*y - 3 = 0) → (x - y = 0) := by
  sorry

#check common_chord_of_circles

end common_chord_of_circles_l2637_263770


namespace waiter_tip_earnings_l2637_263716

theorem waiter_tip_earnings (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : 
  total_customers = 10 →
  non_tipping_customers = 5 →
  tip_amount = 3 →
  (total_customers - non_tipping_customers) * tip_amount = 15 := by
sorry

end waiter_tip_earnings_l2637_263716


namespace max_value_plus_cos_squared_l2637_263733

theorem max_value_plus_cos_squared (x : ℝ) (M : ℝ) : 
  0 ≤ x → x ≤ π / 2 → 
  (∀ y, 0 ≤ y ∧ y ≤ π / 2 → 
    3 * Real.sin y ^ 2 + 8 * Real.sin y * Real.cos y + 9 * Real.cos y ^ 2 ≤ M) →
  (3 * Real.sin x ^ 2 + 8 * Real.sin x * Real.cos x + 9 * Real.cos x ^ 2 = M) →
  M + 100 * Real.cos x ^ 2 = 91 := by
sorry

end max_value_plus_cos_squared_l2637_263733


namespace intersection_points_slope_l2637_263706

theorem intersection_points_slope :
  ∀ (s x y : ℝ), 
    (2 * x + 3 * y = 8 * s + 5) →
    (x + 2 * y = 3 * s + 2) →
    y = -(7/2) * x + 1/2 := by
  sorry

end intersection_points_slope_l2637_263706


namespace complex_modulus_product_l2637_263772

theorem complex_modulus_product : Complex.abs ((10 - 6*I) * (7 + 24*I)) = 25 * Real.sqrt 136 := by
  sorry

end complex_modulus_product_l2637_263772


namespace students_left_l2637_263762

theorem students_left (total : ℕ) (checked_out : ℕ) (h1 : total = 124) (h2 : checked_out = 93) :
  total - checked_out = 31 := by
  sorry

end students_left_l2637_263762


namespace min_sum_squares_l2637_263747

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  ∃ (m : ℝ), m = (1:ℝ)/14 ∧ x^2 + y^2 + z^2 ≥ m ∧ 
  (x^2 + y^2 + z^2 = m ↔ x = (1:ℝ)/14 ∧ y = (1:ℝ)/7 ∧ z = (3:ℝ)/14) :=
by sorry

end min_sum_squares_l2637_263747


namespace abs_value_inequality_l2637_263778

theorem abs_value_inequality (x : ℝ) : 2 ≤ |x - 3| ∧ |x - 3| ≤ 4 ↔ (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7) :=
by sorry

end abs_value_inequality_l2637_263778


namespace shortest_side_of_similar_triangle_l2637_263784

/-- Given two similar right triangles, where the first triangle has a side of 15 and a hypotenuse of 17,
    and the second triangle has a hypotenuse of 102, the shortest side of the second triangle is 48. -/
theorem shortest_side_of_similar_triangle (a b c : ℝ) : 
  a ^ 2 + 15 ^ 2 = 17 ^ 2 → -- First triangle is right-angled with side 15 and hypotenuse 17
  a ≤ 15 → -- a is the shortest side of the first triangle
  ∃ (k : ℝ), k > 0 ∧ k * 17 = 102 ∧ k * a = 48 := by
  sorry

end shortest_side_of_similar_triangle_l2637_263784


namespace p_sufficient_not_necessary_for_not_q_l2637_263785

theorem p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0 → |x| ≤ 3) ∧
  (∃ x : ℝ, |x| ≤ 3 ∧ x^2 + 2*x - 3 > 0) :=
by sorry

end p_sufficient_not_necessary_for_not_q_l2637_263785


namespace smallest_solution_quartic_l2637_263794

theorem smallest_solution_quartic (x : ℝ) : 
  (x^4 - 50*x^2 + 625 = 0) → (∃ y : ℝ, y^4 - 50*y^2 + 625 = 0 ∧ y ≤ x) → x ≥ -5 :=
by sorry

end smallest_solution_quartic_l2637_263794


namespace masked_digits_unique_solution_l2637_263722

def is_valid_pair (d : Nat) : Bool :=
  let product := d * d
  product ≥ 10 ∧ product < 100 ∧ product % 10 ≠ d

def get_last_digit (n : Nat) : Nat :=
  n % 10

theorem masked_digits_unique_solution :
  ∃! (elephant mouse pig panda : Nat),
    elephant ≠ mouse ∧ elephant ≠ pig ∧ elephant ≠ panda ∧
    mouse ≠ pig ∧ mouse ≠ panda ∧
    pig ≠ panda ∧
    is_valid_pair mouse ∧
    get_last_digit (mouse * mouse) = elephant ∧
    elephant = 6 ∧ mouse = 4 ∧ pig = 8 ∧ panda = 1 :=
by sorry

end masked_digits_unique_solution_l2637_263722


namespace stating_dieRollSumWays_l2637_263759

/-- Represents the number of faces on a standard die -/
def diefaces : ℕ := 6

/-- Represents the number of times the die is rolled -/
def numrolls : ℕ := 6

/-- Represents the target sum we're aiming for -/
def targetsum : ℕ := 21

/-- 
Calculates the number of ways to roll a fair six-sided die 'numrolls' times 
such that the sum of the outcomes is 'targetsum'
-/
def numWaysToSum (diefaces numrolls targetsum : ℕ) : ℕ := sorry

/-- 
Theorem stating that the number of ways to roll a fair six-sided die six times 
such that the sum of the outcomes is 21 is equal to 15504
-/
theorem dieRollSumWays : numWaysToSum diefaces numrolls targetsum = 15504 := by sorry

end stating_dieRollSumWays_l2637_263759


namespace enclosed_area_equals_four_l2637_263718

-- Define the functions for the line and curve
def f (x : ℝ) : ℝ := 4 * x
def g (x : ℝ) : ℝ := x^3

-- Define the intersection points
def x₁ : ℝ := 0
def x₂ : ℝ := 2

-- State the theorem
theorem enclosed_area_equals_four :
  (∫ x in x₁..x₂, f x - g x) = 4 := by sorry

end enclosed_area_equals_four_l2637_263718


namespace amy_muffins_l2637_263789

def muffins_series (n : ℕ) : ℕ := n * (n + 1) / 2

theorem amy_muffins :
  let days : ℕ := 5
  let start_muffins : ℕ := 1
  let leftover_muffins : ℕ := 7
  let total_brought := muffins_series days
  total_brought + leftover_muffins = 22 :=
by sorry

end amy_muffins_l2637_263789


namespace binary_operation_equality_l2637_263768

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its binary representation as a list of bits. -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec to_bits (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
    to_bits n

/-- The first binary number in the problem: 110110₂ -/
def num1 : List Bool := [true, true, false, true, true, false]

/-- The second binary number in the problem: 101010₂ -/
def num2 : List Bool := [true, false, true, false, true, false]

/-- The divisor in the problem: 100₂ -/
def divisor : List Bool := [true, false, false]

/-- The expected result: 111001101100₂ -/
def expected_result : List Bool := [true, true, true, false, false, true, true, false, true, true, false, false]

/-- Theorem stating the equality of the binary operation and the expected result -/
theorem binary_operation_equality :
  nat_to_binary ((binary_to_nat num1 * binary_to_nat num2) / binary_to_nat divisor) = expected_result :=
sorry

end binary_operation_equality_l2637_263768


namespace g_zero_iff_a_eq_four_thirds_l2637_263796

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x - 4

-- State the theorem
theorem g_zero_iff_a_eq_four_thirds :
  ∀ a : ℝ, g a = 0 ↔ a = 4 / 3 := by sorry

end g_zero_iff_a_eq_four_thirds_l2637_263796


namespace ellipse_and_line_intersection_l2637_263721

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the condition for any point P on the ellipse
def point_condition (PF1 PF2 : ℝ) : Prop :=
  PF1 + PF2 = 2 * Real.sqrt 2

-- Define the focal distance
def focal_distance : ℝ := 2

-- Define the intersecting line
def intersecting_line (x y t : ℝ) : Prop :=
  x - y + t = 0

-- Define the circle condition for the midpoint of AB
def midpoint_condition (x y : ℝ) : Prop :=
  x^2 + y^2 > 10/9

theorem ellipse_and_line_intersection :
  ∀ (a b : ℝ),
  (∀ (x y : ℝ), ellipse_C x y a b → ∃ (PF1 PF2 : ℝ), point_condition PF1 PF2) →
  (a^2 - b^2 = focal_distance^2) →
  (∀ (x y : ℝ), ellipse_C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ (t : ℝ),
    (∃ (x1 y1 x2 y2 : ℝ),
      ellipse_C x1 y1 a b ∧
      ellipse_C x2 y2 a b ∧
      intersecting_line x1 y1 t ∧
      intersecting_line x2 y2 t ∧
      x1 ≠ x2 ∧
      midpoint_condition ((x1 + x2) / 2) ((y1 + y2) / 2)) →
    (-Real.sqrt 3 < t ∧ t ≤ -Real.sqrt 2) ∨ (Real.sqrt 2 ≤ t ∧ t < Real.sqrt 3)) :=
by sorry

end ellipse_and_line_intersection_l2637_263721


namespace raccoon_lock_problem_l2637_263751

theorem raccoon_lock_problem (first_lock_time second_lock_time both_locks_time : ℕ) :
  second_lock_time = 3 * first_lock_time - 3 →
  both_locks_time = 5 * second_lock_time →
  second_lock_time = 60 →
  first_lock_time = 21 := by
  sorry

end raccoon_lock_problem_l2637_263751


namespace gcd_1887_2091_l2637_263761

theorem gcd_1887_2091 : Nat.gcd 1887 2091 = 51 := by
  sorry

end gcd_1887_2091_l2637_263761


namespace fraction_relation_l2637_263757

theorem fraction_relation (w x y z : ℝ) 
  (h1 : x / y = 5)
  (h2 : y / z = 1 / 2)
  (h3 : z / w = 7)
  (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  w / x = 2 / 35 := by
  sorry

end fraction_relation_l2637_263757


namespace budget_projection_l2637_263713

/-- Given the equation fp - w = 15000, where f = 7 and w = 70 + 210i, prove that p = 2153 + 30i -/
theorem budget_projection (f : ℝ) (w p : ℂ) 
  (eq : f * p - w = 15000)
  (hf : f = 7)
  (hw : w = 70 + 210 * Complex.I) : 
  p = 2153 + 30 * Complex.I := by
  sorry

end budget_projection_l2637_263713


namespace complex_power_sum_l2637_263792

theorem complex_power_sum (w : ℂ) (hw : w^2 - w + 1 = 0) : 
  w^103 + w^104 + w^105 + w^106 + w^107 = -1 := by sorry

end complex_power_sum_l2637_263792


namespace program_output_l2637_263798

def S : ℕ → ℚ
  | 0 => 2
  | n + 1 => 1 / (1 - S n)

theorem program_output : S 2017 = -1 := by
  sorry

end program_output_l2637_263798


namespace pythagorean_triple_in_range_l2637_263777

theorem pythagorean_triple_in_range : 
  ∀ a b c : ℕ, 
    a^2 + b^2 = c^2 → 
    Nat.gcd a (Nat.gcd b c) = 1 → 
    2000 ≤ a ∧ a ≤ 3000 → 
    2000 ≤ b ∧ b ≤ 3000 → 
    2000 ≤ c ∧ c ≤ 3000 → 
    (a, b, c) = (2100, 2059, 2941) :=
by sorry

end pythagorean_triple_in_range_l2637_263777


namespace class_size_proof_l2637_263717

theorem class_size_proof :
  ∃ (n : ℕ), 
    n > 0 ∧
    (n / 2 : ℕ) > 0 ∧
    (n / 4 : ℕ) > 0 ∧
    (n / 7 : ℕ) > 0 ∧
    n - (n / 2) - (n / 4) - (n / 7) < 6 ∧
    n = 28 := by
  sorry

end class_size_proof_l2637_263717


namespace elizabeth_steak_knife_cost_l2637_263786

def steak_knife_cost (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℚ) : ℚ :=
  (sets * cost_per_set) / (sets * knives_per_set)

theorem elizabeth_steak_knife_cost :
  steak_knife_cost 2 4 80 = 20 := by
  sorry

end elizabeth_steak_knife_cost_l2637_263786


namespace a_time_is_ten_l2637_263750

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 80 meters long
  a.speed * a.time = 80 ∧
  b.speed * b.time = 80 ∧
  -- A beats B by 56 meters or 7 seconds
  a.speed * (a.time + 7) = 136 ∧
  b.time = a.time + 7

/-- Theorem stating A's time is 10 seconds -/
theorem a_time_is_ten (a b : Runner) (h : Race a b) : a.time = 10 :=
  sorry

end a_time_is_ten_l2637_263750


namespace geometric_sequence_sum_inequality_l2637_263775

theorem geometric_sequence_sum_inequality (n : ℕ) : 2^n - 1 < 2^n := by
  sorry

end geometric_sequence_sum_inequality_l2637_263775


namespace right_triangle_hypotenuse_and_perimeter_l2637_263756

theorem right_triangle_hypotenuse_and_perimeter :
  ∀ (a b h p : ℝ),
  a = 24 →
  b = 32 →
  h^2 = a^2 + b^2 →
  p = a + b + h →
  h = 40 ∧ p = 96 := by
  sorry

end right_triangle_hypotenuse_and_perimeter_l2637_263756


namespace fish_pond_population_l2637_263734

-- Define the parameters
def initial_tagged : ℕ := 40
def second_catch : ℕ := 50
def tagged_in_second : ℕ := 2

-- Define the theorem
theorem fish_pond_population :
  let total_fish : ℕ := (initial_tagged * second_catch) / tagged_in_second
  total_fish = 1000 := by
  sorry

end fish_pond_population_l2637_263734


namespace remainder_18_pow_63_mod_5_l2637_263737

theorem remainder_18_pow_63_mod_5 : 18^63 % 5 = 2 := by
  sorry

end remainder_18_pow_63_mod_5_l2637_263737


namespace right_triangle_sin_d_l2637_263701

theorem right_triangle_sin_d (D E F : ℝ) (h1 : 0 < D) (h2 : D < π / 2) : 
  5 * Real.sin D = 12 * Real.cos D → Real.sin D = 12 / 13 := by
sorry

end right_triangle_sin_d_l2637_263701


namespace profit_calculation_l2637_263719

/-- Given that the cost price of 30 articles equals the selling price of x articles,
    and the profit is 25%, prove that x = 24. -/
theorem profit_calculation (x : ℝ) 
  (h1 : 30 * cost_price = x * selling_price)
  (h2 : selling_price = 1.25 * cost_price) : 
  x = 24 := by
  sorry

end profit_calculation_l2637_263719


namespace geometry_theorems_l2637_263740

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Axioms for the properties of parallel and perpendicular
axiom parallel_planes_transitive : 
  ∀ (α β γ : Plane), parallel_planes α β → parallel_planes α γ → parallel_planes β γ

axiom perpendicular_parallel_planes : 
  ∀ (m : Line) (α β : Plane), 
    perpendicular_line_plane m α → parallel_line_plane m β → perpendicular_planes α β

-- Theorem to prove
theorem geometry_theorems :
  (∀ (α β γ : Plane), parallel_planes α β → parallel_planes α γ → parallel_planes β γ) ∧
  (∀ (m : Line) (α β : Plane), 
    perpendicular_line_plane m α → parallel_line_plane m β → perpendicular_planes α β) :=
by sorry

end geometry_theorems_l2637_263740


namespace pure_imaginary_condition_l2637_263755

theorem pure_imaginary_condition (m : ℝ) : 
  (Complex.I * (m^2 - 1) = m^2 + m - 2 + Complex.I * (m^2 - 1)) → m = -2 :=
by sorry

end pure_imaginary_condition_l2637_263755


namespace tadpole_fish_difference_l2637_263799

def initial_fish : ℕ := 50
def tadpole_ratio : ℕ := 3
def fish_caught : ℕ := 7
def tadpole_development_ratio : ℚ := 1/2

theorem tadpole_fish_difference : 
  (tadpole_ratio * initial_fish) * tadpole_development_ratio - (initial_fish - fish_caught) = 32 := by
  sorry

end tadpole_fish_difference_l2637_263799


namespace distance_between_points_l2637_263773

theorem distance_between_points : ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = 0 ∧ y1 = 6 ∧ x2 = 8 ∧ y2 = 0 → 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 10 := by
  sorry

end distance_between_points_l2637_263773


namespace shortest_paths_count_julia_paths_count_l2637_263726

theorem shortest_paths_count : Nat → Nat → Nat
| m, n => Nat.choose (m + n) m

theorem julia_paths_count : shortest_paths_count 8 5 = 1287 := by
  sorry

end shortest_paths_count_julia_paths_count_l2637_263726


namespace problem_solution_l2637_263745

def set_A (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a^2 + 1}
def set_B (x : ℝ) : Set ℝ := {0, 1, x}

theorem problem_solution :
  (∀ a : ℝ, -3 ∈ set_A a → a = 0 ∨ a = -1) ∧
  (∀ x : ℝ, x^2 ∈ set_B x ∧ x ≠ 0 ∧ x ≠ 1 → x = -1) :=
by sorry

end problem_solution_l2637_263745


namespace four_identical_differences_l2637_263743

theorem four_identical_differences (S : Finset ℕ) : 
  S.card = 20 → (∀ n ∈ S, n < 70) → 
  ∃ (d : ℕ) (a b c d e f g h : ℕ), 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
    a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ g ≠ h ∧
    b - a = d - c ∧ d - c = f - e ∧ f - e = h - g ∧ h - g = d :=
by sorry

end four_identical_differences_l2637_263743


namespace parametric_to_standard_equation_l2637_263709

theorem parametric_to_standard_equation (x y θ : ℝ) 
  (h1 : x = 1 + 2 * Real.cos θ) 
  (h2 : y = 2 * Real.sin θ) : 
  (x - 1)^2 + y^2 = 4 := by
sorry

end parametric_to_standard_equation_l2637_263709


namespace unique_numbers_l2637_263704

/-- Checks if a three-digit number has distinct digits in ascending order -/
def has_ascending_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a < b ∧ b < c

/-- Checks if a three-digit number has identical digits -/
def has_identical_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a : ℕ), n = 100 * a + 10 * a + a

/-- Checks if all words in the name of a number start with the same letter -/
def name_starts_same_letter (n : ℕ) : Prop :=
  -- This is a placeholder for the actual condition
  n = 147

/-- Checks if all words in the name of a number start with different letters -/
def name_starts_different_letters (n : ℕ) : Prop :=
  -- This is a placeholder for the actual condition
  n = 111

theorem unique_numbers :
  (∃! n : ℕ, has_ascending_digits n ∧ name_starts_same_letter n) ∧
  (∃! n : ℕ, has_identical_digits n ∧ name_starts_different_letters n) :=
sorry

end unique_numbers_l2637_263704


namespace age_problem_l2637_263753

/-- The age problem -/
theorem age_problem (sebastian_age : ℕ) (sister_age : ℕ) (father_age : ℕ) : 
  sebastian_age = 40 →
  sister_age = sebastian_age - 10 →
  (sebastian_age - 5) + (sister_age - 5) = 3 * (father_age - 5) / 4 →
  father_age = 90 := by
sorry

end age_problem_l2637_263753


namespace ball_probability_and_replacement_l2637_263736

/-- Given a bag with red, yellow, and blue balls, this theorem proves:
    1. The initial probability of drawing a red ball.
    2. The number of red balls replaced to achieve a specific probability of drawing a yellow ball. -/
theorem ball_probability_and_replacement 
  (initial_red : ℕ) 
  (initial_yellow : ℕ) 
  (initial_blue : ℕ) 
  (replaced : ℕ) :
  initial_red = 10 → 
  initial_yellow = 2 → 
  initial_blue = 8 → 
  (initial_red : ℚ) / (initial_red + initial_yellow + initial_blue : ℚ) = 1/2 ∧
  (initial_yellow + replaced : ℚ) / (initial_red + initial_yellow + initial_blue : ℚ) = 2/5 →
  replaced = 6 := by
  sorry

end ball_probability_and_replacement_l2637_263736


namespace derivative_at_one_l2637_263730

/-- Given a differentiable function f: ℝ → ℝ where x > 0, 
    if f(x) = 2e^x * f'(1) + 3ln(x), then f'(1) = 3 / (1 - 2e) -/
theorem derivative_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x > 0, f x = 2 * Real.exp x * deriv f 1 + 3 * Real.log x) : 
  deriv f 1 = 3 / (1 - 2 * Real.exp 1) := by
  sorry

end derivative_at_one_l2637_263730


namespace election_votes_total_l2637_263758

theorem election_votes_total (winning_percentage : ℚ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 60 / 100 →
  majority = 1300 →
  winning_percentage * total_votes - (1 - winning_percentage) * total_votes = majority →
  total_votes = 6500 :=
by sorry

end election_votes_total_l2637_263758


namespace circle_C_properties_l2637_263766

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 2)^2 = 9

-- Define the line L where the center of C lies
def line_L (x y : ℝ) : Prop :=
  x + 2*y + 1 = 0

-- Define the line that potentially intersects C
def intersecting_line (a x y : ℝ) : Prop :=
  a*x - y + 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 0)

theorem circle_C_properties :
  -- The circle C passes through M(0,-2) and N(3,1)
  circle_C 0 (-2) ∧ circle_C 3 1 ∧
  -- The center of C lies on line L
  ∃ (cx cy : ℝ), line_L cx cy ∧ ∀ (x y : ℝ), circle_C x y ↔ (x - cx)^2 + (y - cy)^2 = 9 ∧
  -- There's no real a such that the line ax-y+1=0 intersects C at two points
  -- and is perpendicularly bisected by the line through P
  ¬ ∃ (a : ℝ), 
    (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
                          intersecting_line a x₁ y₁ ∧ intersecting_line a x₂ y₂) ∧
    (∃ (mx my : ℝ), circle_C mx my ∧ 
                    (mx - point_P.1) * (x₂ - x₁) + (my - point_P.2) * (y₂ - y₁) = 0 ∧
                    2 * mx = x₁ + x₂ ∧ 2 * my = y₁ + y₂) :=
sorry

end circle_C_properties_l2637_263766


namespace correct_understanding_of_philosophy_l2637_263793

-- Define the characteristics of philosophy
def originatesFromLife (p : Type) : Prop := sorry
def affectsLife (p : Type) : Prop := sorry
def formsSpontaneously (p : Type) : Prop := sorry
def summarizesKnowledge (p : Type) : Prop := sorry

-- Define Yu Wujin's statement
def yuWujinStatement (p : Type) : Prop := sorry

-- Theorem to prove
theorem correct_understanding_of_philosophy (p : Type) :
  yuWujinStatement p ↔ (originatesFromLife p ∧ affectsLife p) :=
sorry

end correct_understanding_of_philosophy_l2637_263793


namespace square_of_real_not_always_positive_l2637_263764

theorem square_of_real_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end square_of_real_not_always_positive_l2637_263764


namespace function_has_extrema_l2637_263749

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + (2*a + 1)*x

-- State the theorem
theorem function_has_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
   (∀ x, f a x ≥ f a x₁) ∧ 
   (∀ x, f a x ≤ f a x₂)) ↔ 
  (a > 1 ∨ a < -1/3) :=
sorry

end function_has_extrema_l2637_263749


namespace chandler_bike_savings_l2637_263708

/-- The number of weeks Chandler needs to save to buy a mountain bike -/
def weeks_to_save (bike_cost : ℕ) (birthday_money : ℕ) (weekly_earnings : ℕ) : ℕ :=
  (bike_cost - birthday_money) / weekly_earnings

theorem chandler_bike_savings : 
  let bike_cost : ℕ := 600
  let grandparents_gift : ℕ := 60
  let aunt_gift : ℕ := 40
  let cousin_gift : ℕ := 20
  let weekly_earnings : ℕ := 20
  let total_birthday_money : ℕ := grandparents_gift + aunt_gift + cousin_gift
  weeks_to_save bike_cost total_birthday_money weekly_earnings = 24 := by
  sorry

#eval weeks_to_save 600 (60 + 40 + 20) 20

end chandler_bike_savings_l2637_263708


namespace max_discarded_apples_l2637_263710

theorem max_discarded_apples (n : ℕ) : ∃ (q : ℕ), n = 7 * q + 6 ∧ 
  ∀ (r : ℕ), r < 6 → ∃ (q' : ℕ), n ≠ 7 * q' + r :=
sorry

end max_discarded_apples_l2637_263710


namespace remainder_theorem_l2637_263767

def f (x : ℝ) : ℝ := 5*x^7 - 3*x^6 - 8*x^5 + 3*x^3 + 5*x^2 - 20

def g (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem :
  ∃ q : ℝ → ℝ, f = fun x ↦ g x * q x + 6910 := by
  sorry

end remainder_theorem_l2637_263767


namespace intersection_A_complement_B_l2637_263763

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 2, 3}

-- Define set B
def B : Finset Nat := {2, 5}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 3} := by
  sorry

end intersection_A_complement_B_l2637_263763


namespace possible_values_of_a_l2637_263723

def A (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A a ⊆ B) ↔ (a = -2 ∨ a = -1) :=
by sorry

end possible_values_of_a_l2637_263723


namespace geometric_sequence_common_ratio_l2637_263754

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h1 : a 3 - 3 * a 2 = 2) 
  (h2 : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 3 := by
  sorry

end geometric_sequence_common_ratio_l2637_263754


namespace inequality_subtraction_l2637_263742

theorem inequality_subtraction (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end inequality_subtraction_l2637_263742


namespace quadratic_equation_roots_l2637_263707

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a > 0) (h2 : c < 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
sorry

end quadratic_equation_roots_l2637_263707


namespace units_digit_of_A_is_one_l2637_263779

-- Define the function for the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the expression for A
def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

-- Theorem statement
theorem units_digit_of_A_is_one : unitsDigit A = 1 := by
  sorry

end units_digit_of_A_is_one_l2637_263779


namespace tangent_line_proofs_l2637_263771

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_proofs :
  let e := Real.exp 1
  -- Tangent line at (e, e^e)
  ∃ (m : ℝ), ∀ x y : ℝ,
    (y = f x) → (x = e ∧ y = f e) →
    (m * (x - e) + f e = y ∧ m * x - y - m * e + f e = 0) →
    (Real.exp e * x - y - Real.exp (e + 1) = 0) ∧
  -- Tangent line from origin
  ∃ (k : ℝ), ∀ x y : ℝ,
    (y = f x) → (y = k * x) →
    (k = f x ∧ k = (f x) / x) →
    (e * x - y = 0) := by
  sorry

end tangent_line_proofs_l2637_263771


namespace bus_problem_l2637_263760

/-- The number of people who got off the bus -/
def people_got_off (initial : ℕ) (final : ℕ) : ℕ := initial - final

/-- Theorem stating that 47 people got off the bus -/
theorem bus_problem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 90) 
  (h2 : final = 43) : 
  people_got_off initial final = 47 := by
  sorry

end bus_problem_l2637_263760


namespace elisa_target_amount_l2637_263741

/-- Elisa's target amount problem -/
theorem elisa_target_amount (current_amount additional_amount : ℕ) 
  (h1 : current_amount = 37)
  (h2 : additional_amount = 16) :
  current_amount + additional_amount = 53 := by
  sorry

end elisa_target_amount_l2637_263741


namespace cube_increase_theorem_l2637_263769

theorem cube_increase_theorem :
  let s : ℝ := 1  -- Initial side length (can be any positive real number)
  let s' : ℝ := 1.2 * s  -- New side length after 20% increase
  let A : ℝ := 6 * s^2  -- Initial surface area
  let V : ℝ := s^3  -- Initial volume
  let A' : ℝ := 6 * s'^2  -- New surface area
  let V' : ℝ := s'^3  -- New volume
  let x : ℝ := (A' - A) / A * 100  -- Percentage increase in surface area
  let y : ℝ := (V' - V) / V * 100  -- Percentage increase in volume
  5 * (y - x) = 144 := by sorry

end cube_increase_theorem_l2637_263769


namespace product_max_value_l2637_263703

theorem product_max_value (x y z u : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ u ≥ 0) 
  (h_constraint : 2*x + x*y + z + y*z*u = 1) : 
  x^2 * y^2 * z^2 * u ≤ 1/512 := by
  sorry

end product_max_value_l2637_263703


namespace inequality_preservation_l2637_263732

theorem inequality_preservation (a b c : ℝ) (h : a < b) (h' : b < 0) : a - c < b - c := by
  sorry

end inequality_preservation_l2637_263732


namespace kayla_driving_years_l2637_263776

/-- The minimum driving age in Kayla's state -/
def minimum_driving_age : ℕ := 18

/-- Kimiko's age -/
def kimiko_age : ℕ := 26

/-- Kayla's current age -/
def kayla_age : ℕ := kimiko_age / 2

/-- The number of years before Kayla can reach the minimum driving age -/
def years_until_driving : ℕ := minimum_driving_age - kayla_age

theorem kayla_driving_years :
  years_until_driving = 5 :=
by sorry

end kayla_driving_years_l2637_263776


namespace flag_design_count_l2637_263705

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  num_flag_designs = 27 :=
sorry

end flag_design_count_l2637_263705


namespace time_per_furniture_piece_l2637_263795

theorem time_per_furniture_piece (chairs tables total_time : ℕ) : 
  chairs = 7 → tables = 3 → total_time = 40 → (chairs + tables) * 4 = total_time := by
  sorry

end time_per_furniture_piece_l2637_263795


namespace prob_one_good_product_prob_one_good_product_proof_l2637_263731

/-- The probability of selecting exactly one good product when randomly selecting
    two products from a set of five products, where three are good and two are defective. -/
theorem prob_one_good_product : ℚ :=
  let total_products : ℕ := 5
  let good_products : ℕ := 3
  let defective_products : ℕ := 2
  let selected_products : ℕ := 2
  3 / 5

/-- Proof that the probability of selecting exactly one good product is 3/5. -/
theorem prob_one_good_product_proof :
  prob_one_good_product = 3 / 5 := by
  sorry

end prob_one_good_product_prob_one_good_product_proof_l2637_263731


namespace subset_union_theorem_l2637_263735

theorem subset_union_theorem (n : ℕ) (X : Finset ℕ) (m : ℕ) 
  (A : Fin m → Finset ℕ) :
  n > 6 →
  X.card = n →
  (∀ i : Fin m, (A i).card = 5) →
  (∀ i j : Fin m, i ≠ j → A i ≠ A j) →
  m > n * (n - 1) * (n - 2) * (n - 3) * (4 * n - 15) / 600 →
  ∃ (i₁ i₂ i₃ i₄ i₅ i₆ : Fin m), 
    i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ < i₅ ∧ i₅ < i₆ ∧
    (A i₁ ∪ A i₂ ∪ A i₃ ∪ A i₄ ∪ A i₅ ∪ A i₆) = X :=
by sorry

end subset_union_theorem_l2637_263735


namespace factorization_equality_l2637_263791

theorem factorization_equality (x : ℝ) : 5*x*(x+2) + 9*(x+2) = (x+2)*(5*x+9) := by
  sorry

end factorization_equality_l2637_263791


namespace cycle_original_price_l2637_263724

/-- Proves that given a cycle sold at a loss of 18% with a selling price of 1148, the original price of the cycle was 1400. -/
theorem cycle_original_price (loss_percentage : ℝ) (selling_price : ℝ) (original_price : ℝ) : 
  loss_percentage = 18 →
  selling_price = 1148 →
  selling_price = (1 - loss_percentage / 100) * original_price →
  original_price = 1400 := by
sorry

end cycle_original_price_l2637_263724


namespace distance_traveled_l2637_263781

/-- Given a speed of 75 km/hr and a time of 4 hours, prove that the distance traveled is 300 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 75) (h2 : time = 4) :
  speed * time = 300 := by
  sorry

end distance_traveled_l2637_263781


namespace S_bounds_l2637_263715

def S : Set ℝ := { y | ∃ x : ℝ, x ≥ 0 ∧ y = (2 * x + 3) / (x + 2) }

theorem S_bounds :
  ∃ (m M : ℝ),
    (∀ y ∈ S, m ≤ y) ∧
    (∀ y ∈ S, y ≤ M) ∧
    m ∈ S ∧
    M ∉ S ∧
    m = 3/2 ∧
    M = 2 := by
  sorry


end S_bounds_l2637_263715


namespace brian_shirts_l2637_263787

theorem brian_shirts (steven_shirts andrew_shirts brian_shirts : ℕ) 
  (h1 : steven_shirts = 4 * andrew_shirts)
  (h2 : andrew_shirts = 6 * brian_shirts)
  (h3 : steven_shirts = 72) : 
  brian_shirts = 3 := by
sorry

end brian_shirts_l2637_263787


namespace largest_multiple_of_8_under_100_l2637_263748

theorem largest_multiple_of_8_under_100 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 100 → n ≤ 96 :=
by
  sorry

end largest_multiple_of_8_under_100_l2637_263748


namespace max_value_expression_l2637_263780

theorem max_value_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 10) :
  Real.sqrt (2 * x + 20) + Real.sqrt (26 - 2 * x) + Real.sqrt (3 * x) ≤ 4 * Real.sqrt 79 ∧
  (x = 10 → Real.sqrt (2 * x + 20) + Real.sqrt (26 - 2 * x) + Real.sqrt (3 * x) = 4 * Real.sqrt 79) := by
  sorry

end max_value_expression_l2637_263780


namespace a_values_l2637_263720

/-- The set of real numbers x such that x^2 - 2x - 8 = 0 -/
def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}

/-- The set of real numbers x such that x^2 + a*x + a^2 - 12 = 0, where a is a parameter -/
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

/-- The set of all possible values for a given the conditions -/
def possible_a : Set ℝ := {a | a < -4 ∨ a = -2 ∨ a ≥ 4}

theorem a_values (h : A ∪ B a = A) : a ∈ possible_a := by
  sorry

end a_values_l2637_263720


namespace five_rooks_on_five_by_five_board_l2637_263738

/-- The number of ways to place n distinct rooks on an n×n chess board
    such that no two rooks share the same row or column -/
def rook_placements (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: The number of ways to place 5 distinct rooks on a 5×5 chess board,
    such that no two rooks share the same row or column, is equal to 5! (120) -/
theorem five_rooks_on_five_by_five_board :
  rook_placements 5 = 120 := by
  sorry

#eval rook_placements 5  -- Should output 120

end five_rooks_on_five_by_five_board_l2637_263738


namespace sum_of_ages_l2637_263739

/-- Given a son who is 27 years old and a woman whose age is three years more
    than twice her son's age, prove that the sum of their ages is 84 years. -/
theorem sum_of_ages (son_age : ℕ) (woman_age : ℕ) : son_age = 27 →
  woman_age = 2 * son_age + 3 → son_age + woman_age = 84 := by
  sorry

end sum_of_ages_l2637_263739


namespace inequality_proof_l2637_263774

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) ≥ 12) ∧
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end inequality_proof_l2637_263774


namespace triangle_expression_simplification_l2637_263782

/-- Given a triangle with side lengths x, y, and z, prove that |x+y-z|-2|y-x-z| = -x + 3y - 3z -/
theorem triangle_expression_simplification
  (x y z : ℝ)
  (hxy : x + y > z)
  (hyz : y + z > x)
  (hxz : x + z > y) :
  |x + y - z| - 2 * |y - x - z| = -x + 3*y - 3*z := by
  sorry

end triangle_expression_simplification_l2637_263782


namespace top_pyramid_volume_calculation_l2637_263765

/-- A right square pyramid with given dimensions -/
structure RightSquarePyramid where
  base_edge : ℝ
  slant_edge : ℝ

/-- The volume of the top portion of a right square pyramid cut by a plane parallel to its base -/
def top_pyramid_volume (p : RightSquarePyramid) (cut_ratio : ℝ) : ℝ :=
  sorry

/-- The main theorem stating the volume of the top portion of the cut pyramid -/
theorem top_pyramid_volume_calculation (p : RightSquarePyramid) 
  (h_base : p.base_edge = 10 * Real.sqrt 2)
  (h_slant : p.slant_edge = 12)
  (h_cut_ratio : cut_ratio = 1/4) :
  top_pyramid_volume p cut_ratio = 84.375 * Real.sqrt 11 := by
  sorry

end top_pyramid_volume_calculation_l2637_263765


namespace injective_function_property_l2637_263711

theorem injective_function_property (f : ℕ → ℕ) :
  (∀ m n : ℕ, m > 0 → n > 0 → f (n * f m) ≤ n * m) →
  Function.Injective f →
  ∀ x : ℕ, f x = x := by
  sorry

end injective_function_property_l2637_263711


namespace haley_concert_spending_l2637_263702

/-- The amount spent on concert tickets -/
def concert_spending (ticket_price : ℕ) (tickets_for_self : ℕ) (extra_tickets : ℕ) : ℕ :=
  ticket_price * (tickets_for_self + extra_tickets)

theorem haley_concert_spending :
  concert_spending 4 3 5 = 32 := by
  sorry

end haley_concert_spending_l2637_263702


namespace polynomial_divisibility_l2637_263714

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 3) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 8)) →
  p = -67/3 ∧ q = -158/3 := by
sorry

end polynomial_divisibility_l2637_263714


namespace gold_coin_distribution_l2637_263783

theorem gold_coin_distribution (x y : ℕ) (h : x^2 - y^2 = 16 * (x - y)) : x + y = 16 := by
  sorry

end gold_coin_distribution_l2637_263783


namespace remainder_problem_l2637_263728

theorem remainder_problem (m : ℤ) : (((8 - m) + (m + 4)) % 5) % 3 = 2 := by
  sorry

end remainder_problem_l2637_263728
