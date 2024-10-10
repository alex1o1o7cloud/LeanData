import Mathlib

namespace tracy_candies_l3237_323773

theorem tracy_candies (x : ℕ) (h1 : x % 4 = 0) 
  (h2 : x / 2 % 2 = 0) 
  (h3 : 4 ≤ x / 2 - 20) (h4 : x / 2 - 20 ≤ 8) 
  (h5 : ∃ (b : ℕ), 2 ≤ b ∧ b ≤ 6 ∧ x / 2 - 20 - b = 4) : x = 48 := by
  sorry

end tracy_candies_l3237_323773


namespace monogram_count_l3237_323772

/-- The number of letters in the alphabet excluding 'A' -/
def n : ℕ := 25

/-- The number of letters we need to choose for middle and last initials -/
def k : ℕ := 2

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of ways to choose two distinct letters from 25 letters in alphabetical order is 300 -/
theorem monogram_count : choose n k = 300 := by sorry

end monogram_count_l3237_323772


namespace quadratic_equation_k_value_l3237_323765

/-- Given a quadratic equation with parameter k, prove that k = 1 under specific conditions -/
theorem quadratic_equation_k_value (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ 
    x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 ∧
    x1 + x2 + 2*x1*x2 = 1) →
  k = 1 :=
by sorry

end quadratic_equation_k_value_l3237_323765


namespace inverse_variation_example_l3237_323798

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_variation_example :
  ∀ x y : ℝ → ℝ,
  VaryInversely x y →
  y 1500 = 0.4 →
  y 3000 = 0.2 := by
sorry

end inverse_variation_example_l3237_323798


namespace equation_solution_l3237_323750

theorem equation_solution :
  let f (x : ℝ) := (7 * x + 3) / (3 * x^2 + 7 * x - 6)
  let g (x : ℝ) := (3 * x) / (3 * x - 2)
  let sol₁ := (-1 + Real.sqrt 10) / 3
  let sol₂ := (-1 - Real.sqrt 10) / 3
  ∀ x : ℝ, x ≠ 2/3 →
    (f x = g x ↔ x = sol₁ ∨ x = sol₂) :=
by sorry

end equation_solution_l3237_323750


namespace banana_arrangements_l3237_323789

def word_length : ℕ := 6
def occurrences : List ℕ := [1, 2, 3]

theorem banana_arrangements :
  (word_length.factorial) / (occurrences.prod) = 60 := by
  sorry

end banana_arrangements_l3237_323789


namespace frog_arrangement_count_l3237_323713

/-- Represents the number of ways to arrange frogs with given constraints -/
def frog_arrangements (total : ℕ) (green : ℕ) (red : ℕ) (blue : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 8 3 4 1 = 576 :=
by
  sorry

end frog_arrangement_count_l3237_323713


namespace physics_class_size_l3237_323744

theorem physics_class_size 
  (total_students : ℕ) 
  (math_only : ℕ) 
  (physics_only : ℕ) 
  (both : ℕ) :
  total_students = 53 →
  both = 7 →
  physics_only + both = 2 * (math_only + both) →
  total_students = math_only + physics_only + both →
  physics_only + both = 40 := by
  sorry

end physics_class_size_l3237_323744


namespace amp_eight_five_plus_ten_l3237_323721

def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem amp_eight_five_plus_ten : (amp 8 5) + 10 = 49 := by
  sorry

end amp_eight_five_plus_ten_l3237_323721


namespace complex_locus_is_ellipse_l3237_323783

theorem complex_locus_is_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (w : ℂ), w = 2 * z + 1 / z →
  (w.re / a) ^ 2 + (w.im / b) ^ 2 = 1 :=
sorry

end complex_locus_is_ellipse_l3237_323783


namespace solve_exponential_system_l3237_323759

theorem solve_exponential_system (x y : ℝ) 
  (h1 : (6 : ℝ) ^ (x + y) = 36)
  (h2 : (6 : ℝ) ^ (x + 5 * y) = 216) :
  x = 7 / 4 := by
  sorry

end solve_exponential_system_l3237_323759


namespace rectangular_floor_shorter_side_l3237_323746

theorem rectangular_floor_shorter_side (floor_length : ℝ) (floor_width : ℝ) 
  (carpet_side : ℝ) (carpet_cost : ℝ) (total_cost : ℝ) :
  floor_length = 10 →
  carpet_side = 2 →
  carpet_cost = 15 →
  total_cost = 225 →
  floor_width * floor_length = (total_cost / carpet_cost) * carpet_side^2 →
  floor_width = 6 := by
sorry

end rectangular_floor_shorter_side_l3237_323746


namespace baseball_cards_cost_l3237_323763

theorem baseball_cards_cost (football_pack_cost : ℝ) (pokemon_pack_cost : ℝ) (total_spent : ℝ)
  (h1 : football_pack_cost = 2.73)
  (h2 : pokemon_pack_cost = 4.01)
  (h3 : total_spent = 18.42) :
  total_spent - (2 * football_pack_cost + pokemon_pack_cost) = 8.95 := by
  sorry

end baseball_cards_cost_l3237_323763


namespace possible_y_values_l3237_323709

-- Define the relationship between x and y
def relation (x y : ℝ) : Prop := x^2 = y - 5

-- Theorem statement
theorem possible_y_values :
  (∃ y : ℝ, relation (-7) y ∧ y = 54) ∧
  (∃ y : ℝ, relation 2 y ∧ y = 9) := by
  sorry

end possible_y_values_l3237_323709


namespace H_range_l3237_323737

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem H_range : Set.range H = Set.Icc (-5) 5 := by sorry

end H_range_l3237_323737


namespace ratio_equality_sometimes_l3237_323735

/-- An isosceles triangle with side lengths A and base B -/
structure IsoscelesTriangle where
  A : ℝ
  B : ℝ
  h : ℝ  -- height
  K₁ : ℝ  -- area
  β : ℝ  -- base angle
  h_eq : h = Real.sqrt (A^2 - (B/2)^2)
  K₁_eq : K₁ = (1/2) * B * h
  B_ne_A : B ≠ A

/-- An equilateral triangle with side length a -/
structure EquilateralTriangle where
  a : ℝ
  p : ℝ  -- perimeter
  k₁ : ℝ  -- area
  α : ℝ  -- angle
  p_eq : p = 3 * a
  k₁_eq : k₁ = (a^2 * Real.sqrt 3) / 4
  α_eq : α = π / 3

/-- The main theorem stating that the ratio equality holds sometimes but not always -/
theorem ratio_equality_sometimes (iso : IsoscelesTriangle) (equi : EquilateralTriangle)
    (h_eq : iso.A = equi.a) :
    ∃ (iso₁ : IsoscelesTriangle) (equi₁ : EquilateralTriangle),
      iso₁.h / equi₁.p = iso₁.K₁ / equi₁.k₁ ∧
    ∃ (iso₂ : IsoscelesTriangle) (equi₂ : EquilateralTriangle),
      iso₂.h / equi₂.p ≠ iso₂.K₁ / equi₂.k₁ := by
  sorry

end ratio_equality_sometimes_l3237_323735


namespace club_members_count_l3237_323780

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 8

/-- The total cost of apparel for all members in dollars -/
def total_cost : ℕ := 4440

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 1

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + tshirt_additional_cost

/-- The cost of apparel for one member in dollars -/
def cost_per_member : ℕ := sock_cost * socks_per_member + tshirt_cost * tshirts_per_member

/-- The number of members in the club -/
def club_members : ℕ := total_cost / cost_per_member

theorem club_members_count : club_members = 130 := by
  sorry

end club_members_count_l3237_323780


namespace dolphins_score_l3237_323790

theorem dolphins_score (total_points sharks_points dolphins_points : ℕ) : 
  total_points = 36 →
  sharks_points = dolphins_points + 12 →
  sharks_points + dolphins_points = total_points →
  dolphins_points = 12 := by
sorry

end dolphins_score_l3237_323790


namespace polynomial_value_theorem_l3237_323745

/-- Given a polynomial function g(x) = px^4 + qx^3 + rx^2 + sx + t,
    if g(-3) = 9, then 16p - 8q + 4r - 2s + t = -9 -/
theorem polynomial_value_theorem (p q r s t : ℝ) :
  let g : ℝ → ℝ := λ x => p * x^4 + q * x^3 + r * x^2 + s * x + t
  g (-3) = 9 → 16 * p - 8 * q + 4 * r - 2 * s + t = -9 := by
  sorry

end polynomial_value_theorem_l3237_323745


namespace parallelogram_side_ge_altitude_l3237_323786

/-- A parallelogram with side lengths and altitudes. -/
structure Parallelogram where
  side_a : ℝ
  side_b : ℝ
  altitude_a : ℝ
  altitude_b : ℝ
  side_a_pos : 0 < side_a
  side_b_pos : 0 < side_b
  altitude_a_pos : 0 < altitude_a
  altitude_b_pos : 0 < altitude_b

/-- 
Theorem: For any parallelogram, there exists a side length that is 
greater than or equal to the altitude perpendicular to that side.
-/
theorem parallelogram_side_ge_altitude (p : Parallelogram) :
  (p.side_a ≥ p.altitude_a) ∨ (p.side_b ≥ p.altitude_b) := by
  sorry


end parallelogram_side_ge_altitude_l3237_323786


namespace adjacent_same_face_exists_l3237_323776

/-- Represents the face of a coin -/
inductive CoinFace
| Heads
| Tails

/-- Represents a circular arrangement of coins -/
def CoinCircle := List CoinFace

/-- Checks if two adjacent coins have the same face -/
def hasAdjacentSameFace (circle : CoinCircle) : Prop :=
  ∃ i, (circle.get? i = circle.get? ((i + 1) % circle.length))

/-- Theorem: Any arrangement of 11 coins in a circle always has at least one pair of adjacent coins with the same face -/
theorem adjacent_same_face_exists (circle : CoinCircle) (h : circle.length = 11) :
  hasAdjacentSameFace circle :=
sorry

end adjacent_same_face_exists_l3237_323776


namespace cab_driver_income_l3237_323762

theorem cab_driver_income (income2 income3 income4 income5 avg_income : ℕ)
  (h1 : income2 = 150)
  (h2 : income3 = 750)
  (h3 : income4 = 200)
  (h4 : income5 = 600)
  (h5 : avg_income = 400)
  (h6 : ∃ income1 : ℕ, (income1 + income2 + income3 + income4 + income5) / 5 = avg_income) :
  ∃ income1 : ℕ, income1 = 300 ∧ (income1 + income2 + income3 + income4 + income5) / 5 = avg_income :=
by
  sorry

end cab_driver_income_l3237_323762


namespace three_planes_solutions_two_planes_one_point_solutions_one_plane_two_points_solutions_three_points_solutions_l3237_323708

-- Define the basic types
structure Point
structure Plane

-- Define the distance function
def distance (p : Point) (x : Point ⊕ Plane) : ℝ := sorry

-- Define the function to count solutions
def countSolutions (objects : List (Point ⊕ Plane)) (d : ℝ) : ℕ := sorry

-- Theorem for case (a)
theorem three_planes_solutions (p1 p2 p3 : Plane) (d : ℝ) :
  countSolutions [Sum.inr p1, Sum.inr p2, Sum.inr p3] d = 8 := sorry

-- Theorem for case (b)
theorem two_planes_one_point_solutions (p1 p2 : Plane) (pt : Point) (d : ℝ) :
  countSolutions [Sum.inr p1, Sum.inr p2, Sum.inl pt] d = 8 := sorry

-- Theorem for case (c)
theorem one_plane_two_points_solutions (p : Plane) (pt1 pt2 : Point) (d : ℝ) :
  countSolutions [Sum.inr p, Sum.inl pt1, Sum.inl pt2] d = 4 := sorry

-- Theorem for case (d)
theorem three_points_solutions (pt1 pt2 pt3 : Point) (d : ℝ) :
  let n := countSolutions [Sum.inl pt1, Sum.inl pt2, Sum.inl pt3] d
  n = 0 ∨ n = 1 ∨ n = 2 := sorry

end three_planes_solutions_two_planes_one_point_solutions_one_plane_two_points_solutions_three_points_solutions_l3237_323708


namespace square_side_increase_l3237_323769

theorem square_side_increase (s : ℝ) (h : s > 0) :
  let new_area := s^2 * (1 + 0.3225)
  let new_side := s * (1 + 0.15)
  new_side^2 = new_area := by sorry

end square_side_increase_l3237_323769


namespace work_hours_first_scenario_l3237_323722

/-- Represents the work rate of a person -/
structure WorkRate where
  rate : ℝ

/-- Represents a work scenario -/
structure WorkScenario where
  men : ℕ
  women : ℕ
  hours : ℝ
  days : ℝ

/-- The theorem to prove -/
theorem work_hours_first_scenario 
  (man_rate : WorkRate)
  (woman_rate : WorkRate)
  (scenario1 : WorkScenario)
  (scenario2 : WorkScenario)
  (scenario3 : WorkScenario)
  (h1 : scenario1.men = 2 ∧ scenario1.women = 3 ∧ scenario1.days = 5)
  (h2 : scenario2.men = 4 ∧ scenario2.women = 4 ∧ scenario2.hours = 3 ∧ scenario2.days = 7)
  (h3 : scenario3.men = 7 ∧ scenario3.hours = 4 ∧ scenario3.days = 5.000000000000001)
  (h4 : (scenario1.men : ℝ) * man_rate.rate * scenario1.hours * scenario1.days + 
        (scenario1.women : ℝ) * woman_rate.rate * scenario1.hours * scenario1.days = 1)
  (h5 : (scenario2.men : ℝ) * man_rate.rate * scenario2.hours * scenario2.days + 
        (scenario2.women : ℝ) * woman_rate.rate * scenario2.hours * scenario2.days = 1)
  (h6 : (scenario3.men : ℝ) * man_rate.rate * scenario3.hours * scenario3.days = 1) :
  scenario1.hours = 7 := by
  sorry


end work_hours_first_scenario_l3237_323722


namespace six_digit_multiple_of_three_l3237_323714

theorem six_digit_multiple_of_three : ∃ (n : ℕ), 325473 = 3 * n := by
  sorry

end six_digit_multiple_of_three_l3237_323714


namespace sqrt_fifty_minus_sqrt_thirtytwo_equals_sqrt_two_l3237_323706

theorem sqrt_fifty_minus_sqrt_thirtytwo_equals_sqrt_two :
  Real.sqrt 50 - Real.sqrt 32 = Real.sqrt 2 := by sorry

end sqrt_fifty_minus_sqrt_thirtytwo_equals_sqrt_two_l3237_323706


namespace gas_refill_proof_l3237_323717

def gas_problem (initial_gas tank_capacity gas_to_store gas_to_doctor : ℕ) : Prop :=
  let remaining_gas := initial_gas - gas_to_store - gas_to_doctor
  tank_capacity - remaining_gas = tank_capacity - (initial_gas - gas_to_store - gas_to_doctor)

theorem gas_refill_proof (initial_gas tank_capacity gas_to_store gas_to_doctor : ℕ) 
  (h1 : initial_gas ≥ gas_to_store + gas_to_doctor)
  (h2 : tank_capacity ≥ initial_gas) :
  gas_problem initial_gas tank_capacity gas_to_store gas_to_doctor :=
by
  sorry

#check gas_refill_proof 10 12 6 2

end gas_refill_proof_l3237_323717


namespace polynomial_expansion_l3237_323782

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 2 * x - 3) * (3 * x^3 - x^2) = 15 * x^5 + x^4 - 11 * x^3 + 3 * x^2 := by
  sorry

end polynomial_expansion_l3237_323782


namespace least_positive_integer_satisfying_conditions_l3237_323758

theorem least_positive_integer_satisfying_conditions : ∃ (N : ℕ), 
  (N > 1) ∧ 
  (∃ (a : ℕ), a > 0 ∧ N = a * (2 * a - 1)) ∧ 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 10 → (List.sum (List.range (N - 1))) % k = 0) ∧
  (∀ (M : ℕ), M > 1 ∧ M < N → 
    (∃ (b : ℕ), b > 0 ∧ M = b * (2 * b - 1)) → 
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 10 → (List.sum (List.range (M - 1))) % k = 0) → False) ∧
  N = 2016 :=
by sorry

end least_positive_integer_satisfying_conditions_l3237_323758


namespace decagon_triangles_l3237_323774

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- Theorem stating that the number of triangles formed by the vertices of a regular decagon is 120 -/
theorem decagon_triangles : trianglesInDecagon = 120 := by
  sorry

end decagon_triangles_l3237_323774


namespace cubic_identity_l3237_323727

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end cubic_identity_l3237_323727


namespace max_value_of_f_l3237_323788

noncomputable def f (a b x : ℝ) : ℝ := (4 - x^2) * (a * x^2 + b * x + 5)

theorem max_value_of_f (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-3 - x)) →
  (∃ x : ℝ, ∀ y : ℝ, f a b y ≤ f a b x) ∧
  (∃ x : ℝ, f a b x = 36) :=
sorry

end max_value_of_f_l3237_323788


namespace parabola_intercepts_l3237_323724

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- Theorem statement
theorem parabola_intercepts :
  -- There is exactly one x-intercept at x = 3
  (∃! x : ℝ, x = 3 ∧ ∃ y : ℝ, parabola y = x) ∧
  -- There are exactly two y-intercepts
  (∃! y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ 
    parabola y₁ = 0 ∧ parabola y₂ = 0 ∧
    y₁ = (1 + Real.sqrt 10) / 3 ∧
    y₂ = (1 - Real.sqrt 10) / 3) :=
by sorry

end parabola_intercepts_l3237_323724


namespace least_positive_integer_multiple_47_l3237_323732

def is_multiple (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

theorem least_positive_integer_multiple_47 :
  ∃! x : ℕ+, (x : ℤ) = 5 ∧ 
  (∀ y : ℕ+, y < x → ¬ is_multiple ((2 * y : ℤ)^2 + 2 * 37 * (2 * y) + 37^2) 47) ∧
  is_multiple ((2 * x : ℤ)^2 + 2 * 37 * (2 * x) + 37^2) 47 :=
sorry

end least_positive_integer_multiple_47_l3237_323732


namespace least_positive_y_l3237_323784

theorem least_positive_y (x y : ℤ) : 
  (∃ (k : ℤ), 0 < 24 * x + k * y ∧ ∀ (m : ℤ), 0 < 24 * x + m * y → 24 * x + k * y ≤ 24 * x + m * y) ∧
  (∀ (n : ℤ), 0 < 24 * x + n * y → 4 ≤ 24 * x + n * y) →
  y = 4 ∨ y = -4 :=
sorry

end least_positive_y_l3237_323784


namespace percentage_calculation_l3237_323723

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := by sorry

end percentage_calculation_l3237_323723


namespace circle_bisecting_two_circles_l3237_323711

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 1
def C2 (x y : ℝ) : Prop := (x - 6)^2 + (y + 6)^2 = 9

-- Define a circle C with center (a, 0) and radius r
def C (x y a r : ℝ) : Prop := (x - a)^2 + y^2 = r^2

-- Define the property of C bisecting C1 and C2
def bisects (a r : ℝ) : Prop :=
  ∀ x y : ℝ, C1 x y → (C x y a r ∨ C x y a r)
  ∧ ∀ x y : ℝ, C2 x y → (C x y a r ∨ C x y a r)

-- Theorem statement
theorem circle_bisecting_two_circles :
  ∀ a r : ℝ, bisects a r → C x y 0 9 :=
sorry

end circle_bisecting_two_circles_l3237_323711


namespace september_1_2017_is_friday_l3237_323705

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

def march_19_2017 : Date :=
  { year := 2017, month := 3, day := 19 }

def september_1_2017 : Date :=
  { year := 2017, month := 9, day := 1 }

/-- Returns the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek :=
  sorry

/-- Calculates the number of days between two dates -/
def daysBetween (d1 d2 : Date) : Nat :=
  sorry

theorem september_1_2017_is_friday :
  dayOfWeek march_19_2017 = DayOfWeek.Sunday →
  dayOfWeek september_1_2017 = DayOfWeek.Friday :=
by
  sorry

#check september_1_2017_is_friday

end september_1_2017_is_friday_l3237_323705


namespace frequency_calculation_l3237_323726

/-- Given a sample capacity and a frequency rate, calculate the frequency of a group of samples. -/
def calculate_frequency (sample_capacity : ℕ) (frequency_rate : ℚ) : ℚ :=
  frequency_rate * sample_capacity

/-- Theorem: Given a sample capacity of 32 and a frequency rate of 0.125, the frequency is 4. -/
theorem frequency_calculation :
  let sample_capacity : ℕ := 32
  let frequency_rate : ℚ := 1/8
  calculate_frequency sample_capacity frequency_rate = 4 := by
sorry

end frequency_calculation_l3237_323726


namespace drama_club_subject_distribution_l3237_323787

theorem drama_club_subject_distribution (total : ℕ) (math physics chem : ℕ) 
  (math_physics math_chem physics_chem : ℕ) (all_three : ℕ) :
  total = 100 ∧ 
  math = 50 ∧ 
  physics = 40 ∧ 
  chem = 30 ∧ 
  math_physics = 20 ∧ 
  physics_chem = 10 ∧ 
  all_three = 5 →
  total - (math + physics + chem - math_physics - physics_chem - math_chem + all_three) = 20 :=
by sorry

end drama_club_subject_distribution_l3237_323787


namespace hyperbola_distance_inequality_l3237_323710

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the left focus
def left_focus : ℝ × ℝ := sorry

-- Define a point on the right branch of the hyperbola
def right_branch_point (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2 ∧ P.1 > 0

-- State the theorem
theorem hyperbola_distance_inequality 
  (P₁ P₂ : ℝ × ℝ) 
  (h₁ : right_branch_point P₁) 
  (h₂ : right_branch_point P₂) : 
  dist left_focus P₁ + dist left_focus P₂ - dist P₁ P₂ ≥ 8 :=
sorry

end hyperbola_distance_inequality_l3237_323710


namespace reflected_ray_equation_l3237_323791

/-- The line on which the reflection occurs -/
def reflection_line (x y : ℝ) : Prop := 8 * x + 6 * y = 25

/-- The point through which the reflected ray passes -/
def reflection_point : ℝ × ℝ := (-4, 3)

/-- The origin point from which the incident ray starts -/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem stating that the reflected ray has the equation y = 3 -/
theorem reflected_ray_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (∃ (t : ℝ), reflection_line ((1 - t) * origin.1 + t * x) ((1 - t) * origin.2 + t * y)) →
    (∃ (s : ℝ), x = (1 - s) * reflection_point.1 + s * m ∧ 
                y = (1 - s) * reflection_point.2 + s * 3) :=
sorry

end reflected_ray_equation_l3237_323791


namespace council_vote_difference_l3237_323736

theorem council_vote_difference (total_members : ℕ) 
  (initial_for initial_against : ℕ) 
  (revote_for revote_against : ℕ) : 
  total_members = 500 →
  initial_for + initial_against = total_members →
  initial_against > initial_for →
  revote_for + revote_against = total_members →
  revote_for - revote_against = 3 * (initial_against - initial_for) →
  revote_for = (13 * initial_against) / 12 →
  revote_for - initial_for = 40 := by
sorry

end council_vote_difference_l3237_323736


namespace sunlovers_happy_days_l3237_323775

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2*D*(R^2 + 4) - 2*R*(D^2 + 4) ≥ 0 := by
  sorry

end sunlovers_happy_days_l3237_323775


namespace parabola_translation_correct_l3237_323770

/-- Represents a parabola in the form y = a(x - h)² + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The original parabola y = 2x² --/
def original_parabola : Parabola := { a := 2, h := 0, k := 0 }

/-- The transformed parabola y = 2(x+4)² + 1 --/
def transformed_parabola : Parabola := { a := 2, h := -4, k := 1 }

/-- Represents a translation in 2D space --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The translation that should transform the original parabola to the transformed parabola --/
def correct_translation : Translation := { dx := -4, dy := 1 }

/-- Applies a translation to a parabola --/
def apply_translation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a, h := p.h - t.dx, k := p.k + t.dy }

theorem parabola_translation_correct :
  apply_translation original_parabola correct_translation = transformed_parabola :=
sorry

end parabola_translation_correct_l3237_323770


namespace no_quaint_two_digit_integers_l3237_323777

def is_quaint (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ 
  ∃ (a b : ℕ), n = 10 * a + b ∧ a > 0 ∧ b < 10 ∧ n = a + b^3

theorem no_quaint_two_digit_integers : ¬∃ (n : ℕ), is_quaint n := by
  sorry

end no_quaint_two_digit_integers_l3237_323777


namespace total_grain_calculation_l3237_323768

/-- The amount of grain in kilograms transported from the first warehouse. -/
def transported : ℕ := 2500

/-- The amount of grain in kilograms in the second warehouse. -/
def second_warehouse : ℕ := 50200

/-- The total amount of grain in kilograms in both warehouses. -/
def total_grain : ℕ := second_warehouse + (second_warehouse + transported)

theorem total_grain_calculation :
  total_grain = 102900 :=
by sorry

end total_grain_calculation_l3237_323768


namespace log_product_equals_24_l3237_323749

theorem log_product_equals_24 :
  Real.log 9 / Real.log 2 * (Real.log 16 / Real.log 3) * (Real.log 27 / Real.log 7) = 24 := by
  sorry

end log_product_equals_24_l3237_323749


namespace two_digit_number_puzzle_l3237_323718

theorem two_digit_number_puzzle :
  ∃! n : ℕ,
    n ≥ 10 ∧ n < 100 ∧
    (n / 10 + n % 10 = 8) ∧
    (n - 36 = (n % 10) * 10 + (n / 10)) :=
by sorry

end two_digit_number_puzzle_l3237_323718


namespace teacher_weight_l3237_323799

theorem teacher_weight (num_students : ℕ) (student_avg_weight : ℝ) (avg_increase : ℝ) :
  num_students = 24 →
  student_avg_weight = 35 →
  avg_increase = 0.4 →
  let total_student_weight := num_students * student_avg_weight
  let new_avg := student_avg_weight + avg_increase
  let total_weight_with_teacher := new_avg * (num_students + 1)
  total_weight_with_teacher - total_student_weight = 45 := by
  sorry

end teacher_weight_l3237_323799


namespace all_yarns_are_xants_and_wooks_l3237_323740

-- Define the sets
variable (Zelm Xant Yarn Wook : Type)

-- Define the conditions
variable (zelm_xant : Zelm → Xant)
variable (yarn_zelm : Yarn → Zelm)
variable (xant_wook : Xant → Wook)

-- Theorem to prove
theorem all_yarns_are_xants_and_wooks :
  (∀ y : Yarn, ∃ x : Xant, zelm_xant (yarn_zelm y) = x) ∧
  (∀ y : Yarn, ∃ w : Wook, xant_wook (zelm_xant (yarn_zelm y)) = w) :=
sorry

end all_yarns_are_xants_and_wooks_l3237_323740


namespace ratio_sum_difference_l3237_323797

theorem ratio_sum_difference (a b : ℝ) (h1 : a / b = 3 / 8) (h2 : a + b = 44) : b - a = 20 := by
  sorry

end ratio_sum_difference_l3237_323797


namespace specific_quilt_shaded_fraction_l3237_323725

/-- Represents a square quilt composed of smaller squares -/
structure Quilt :=
  (side_length : ℕ)
  (shaded_triangles : ℕ)
  (shaded_squares : ℕ)

/-- Calculates the fraction of a quilt that is shaded -/
def shaded_fraction (q : Quilt) : ℚ :=
  let total_area : ℚ := (q.side_length * q.side_length : ℚ)
  let shaded_area : ℚ := (q.shaded_squares : ℚ) + (q.shaded_triangles : ℚ) / 2
  shaded_area / total_area

/-- Theorem stating that for a specific quilt configuration, the shaded fraction is 5/18 -/
theorem specific_quilt_shaded_fraction :
  let q := Quilt.mk 3 3 1
  shaded_fraction q = 5 / 18 := by
  sorry

end specific_quilt_shaded_fraction_l3237_323725


namespace largest_number_in_ratio_l3237_323779

theorem largest_number_in_ratio (a b c d : ℚ) : 
  a / b = -3/2 →
  b / c = 4/5 →
  c / d = -2/3 →
  a + b + c + d = 1344 →
  max a (max b (max c d)) = 40320 := by
sorry

end largest_number_in_ratio_l3237_323779


namespace min_value_sum_squares_l3237_323793

theorem min_value_sum_squares (y₁ y₂ y₃ : ℝ) 
  (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 3*y₁ + 2*y₂ + y₃ = 90) : 
  y₁^2 + 4*y₂^2 + 9*y₃^2 ≥ 4050/7 ∧ 
  ∃ y₁' y₂' y₃', y₁'^2 + 4*y₂'^2 + 9*y₃'^2 = 4050/7 ∧ 
                 y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 
                 3*y₁' + 2*y₂' + y₃' = 90 :=
by sorry

end min_value_sum_squares_l3237_323793


namespace J_specific_value_l3237_323764

/-- Definition of J function -/
def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

/-- Theorem: J(3, 3/4, 4) equals 259/48 -/
theorem J_specific_value : J 3 (3/4) 4 = 259/48 := by
  sorry

/-- Lemma: Relationship between a, b, and c -/
lemma abc_relationship (a b c k : ℚ) (hk : k ≠ 0) : 
  b = a / k ∧ c = k * b → J a b c = J a (a / k) (k * (a / k)) := by
  sorry

end J_specific_value_l3237_323764


namespace x_intercept_of_line_l3237_323707

/-- The x-intercept of the line 2x + 3y = 6 is 3 -/
theorem x_intercept_of_line (x y : ℝ) : 2 * x + 3 * y = 6 → y = 0 → x = 3 := by
  sorry

end x_intercept_of_line_l3237_323707


namespace linear_function_properties_l3237_323712

def f (x : ℝ) : ℝ := -2 * x + 3

theorem linear_function_properties :
  (f 1 = 1) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ f x = y) ∧
  (f⁻¹ 0 ≠ 0) ∧
  (∀ (x1 x2 : ℝ), x1 < x2 → f x1 > f x2) :=
by sorry

end linear_function_properties_l3237_323712


namespace bird_count_l3237_323703

theorem bird_count (total_heads : ℕ) (total_legs : ℕ) (bird_legs : ℕ) (mammal_legs : ℕ) (insect_legs : ℕ) :
  total_heads = 300 →
  total_legs = 1112 →
  bird_legs = 2 →
  mammal_legs = 4 →
  insect_legs = 6 →
  ∃ (birds mammals insects : ℕ),
    birds + mammals + insects = total_heads ∧
    birds * bird_legs + mammals * mammal_legs + insects * insect_legs = total_legs ∧
    birds = 122 :=
by sorry

end bird_count_l3237_323703


namespace min_lines_8x8_grid_l3237_323728

/-- Represents a grid with points at the center of each square -/
structure Grid :=
  (size : ℕ)
  (points : ℕ)

/-- Calculates the minimum number of lines needed to separate all points in a grid -/
def min_lines (g : Grid) : ℕ :=
  2 * (g.size - 1)

/-- Theorem: For an 8x8 grid with 64 points, the minimum number of lines to separate all points is 14 -/
theorem min_lines_8x8_grid :
  let g : Grid := ⟨8, 64⟩
  min_lines g = 14 := by
  sorry

end min_lines_8x8_grid_l3237_323728


namespace p_sufficient_for_q_l3237_323754

theorem p_sufficient_for_q : ∀ (x y : ℝ),
  (x - 1)^2 + (y - 1)^2 ≤ 2 →
  y ≥ x - 1 ∧ y ≥ 1 - x ∧ y ≤ 1 := by
  sorry

end p_sufficient_for_q_l3237_323754


namespace ice_skating_given_skiing_l3237_323795

-- Define the probability space
variable (Ω : Type) [MeasurableSpace Ω] [Fintype Ω] (P : Measure Ω)

-- Define events
variable (A B : Set Ω)

-- Define the probabilities
variable (hA : P A = 0.6)
variable (hB : P B = 0.5)
variable (hAorB : P (A ∪ B) = 0.7)

-- Define the theorem
theorem ice_skating_given_skiing :
  P (A ∩ B) / P B = 0.8 :=
sorry

end ice_skating_given_skiing_l3237_323795


namespace f_derivative_at_neg_one_l3237_323738

noncomputable def f' (x : ℝ) : ℝ := 2 * Real.exp x / (Real.exp x - 1)

theorem f_derivative_at_neg_one :
  let f (x : ℝ) := (f' (-1)) * Real.exp x - x^2
  (deriv f) (-1) = 2 * Real.exp 1 / (Real.exp 1 - 1) :=
sorry

end f_derivative_at_neg_one_l3237_323738


namespace unique_positive_integers_sum_l3237_323739

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 2 + 5 / 2)

theorem unique_positive_integers_sum (d e f : ℕ+) :
  y^50 = 2*y^48 + 6*y^46 + 5*y^44 - y^25 + (d:ℝ)*y^21 + (e:ℝ)*y^19 + (f:ℝ)*y^15 →
  d + e + f = 98 :=
by sorry

end unique_positive_integers_sum_l3237_323739


namespace arithmetic_sequence_theorem_l3237_323702

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 15th term of the sequence is 15 -/
def Term15Is15 (a : ℕ → ℝ) : Prop := a 15 = 15

/-- The 16th term of the sequence is 21 -/
def Term16Is21 (a : ℕ → ℝ) : Prop := a 16 = 21

/-- The 3rd term of the sequence is -57 -/
def Term3IsNeg57 (a : ℕ → ℝ) : Prop := a 3 = -57

theorem arithmetic_sequence_theorem (a : ℕ → ℝ) :
  ArithmeticSequence a → Term15Is15 a → Term16Is21 a → Term3IsNeg57 a := by
  sorry

end arithmetic_sequence_theorem_l3237_323702


namespace solution_of_system_l3237_323792

theorem solution_of_system (α β : ℝ) : 
  (∃ (n k : ℤ), (α = π/6 ∨ α = -π/6) ∧ α = α + 2*π*n ∧ 
                 (β = π/4 ∨ β = -π/4) ∧ β = β + 2*π*k) ∨
  (∃ (n k : ℤ), (α = π/4 ∨ α = -π/4) ∧ α = α + 2*π*n ∧ 
                 (β = π/6 ∨ β = -π/6) ∧ β = β + 2*π*k) :=
by sorry

end solution_of_system_l3237_323792


namespace tangent_line_to_parabola_l3237_323767

theorem tangent_line_to_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d ∧ y^2 = 12 * x ∧ 
   ∀ x' y' : ℝ, y' = 3 * x' + d → y'^2 ≥ 12 * x') → 
  d = 1 := by
sorry

end tangent_line_to_parabola_l3237_323767


namespace sandy_shirt_cost_l3237_323700

/-- The amount Sandy spent on clothes -/
def total_spent : ℝ := 33.56

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℝ := 13.99

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℝ := 7.43

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℝ := total_spent - shorts_cost - jacket_cost

theorem sandy_shirt_cost : shirt_cost = 12.14 := by
  sorry

end sandy_shirt_cost_l3237_323700


namespace smallest_coloring_number_l3237_323741

/-- A coloring of positive integers -/
def Coloring (k : ℕ) := ℕ+ → Fin k

/-- A function from positive integers to positive integers -/
def IntegerFunction := ℕ+ → ℕ+

/-- Condition 1: For all n, m of the same color, f(n+m) = f(n) + f(m) -/
def SameColorAdditive (c : Coloring k) (f : IntegerFunction) : Prop :=
  ∀ n m : ℕ+, c n = c m → f (n + m) = f n + f m

/-- Condition 2: There exist n, m such that f(n+m) ≠ f(n) + f(m) -/
def ExistsNonAdditive (f : IntegerFunction) : Prop :=
  ∃ n m : ℕ+, f (n + m) ≠ f n + f m

/-- The main theorem statement -/
theorem smallest_coloring_number :
  (∃ k : ℕ+, ∃ c : Coloring k, ∃ f : IntegerFunction,
    SameColorAdditive c f ∧ ExistsNonAdditive f) ∧
  (∀ k : ℕ+, k < 3 →
    ¬∃ c : Coloring k, ∃ f : IntegerFunction,
      SameColorAdditive c f ∧ ExistsNonAdditive f) :=
sorry

end smallest_coloring_number_l3237_323741


namespace opposite_points_on_number_line_l3237_323748

theorem opposite_points_on_number_line (A B : ℝ) :
  A < B →  -- A is to the left of B
  A = -B →  -- A and B are opposite numbers
  B - A = 6.4 →  -- The distance between A and B is 6.4
  A = -3.2 ∧ B = 3.2 := by
  sorry

end opposite_points_on_number_line_l3237_323748


namespace largest_nice_sequence_l3237_323743

/-- A sequence is nice if it satisfies the given conditions -/
def IsNice (a : ℕ → ℝ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ 
  a 0 + a 1 = -1 / n ∧ 
  ∀ k : ℕ, k ≥ 1 → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) + a (k+1)

/-- The largest N for which a nice sequence of length N+1 exists is equal to n -/
theorem largest_nice_sequence (n : ℕ) : 
  n ≥ 1 → 
  (∃ (N : ℕ) (a : ℕ → ℝ), IsNice a n ∧ N = n) ∧ 
  (∀ (M : ℕ) (a : ℕ → ℝ), M > n → ¬ IsNice a n) :=
sorry

end largest_nice_sequence_l3237_323743


namespace number_comparison_l3237_323766

theorem number_comparison (A B : ℝ) (h : A = B + B / 4) : 
  B = A - A / 5 ∧ B ≠ A - A / 4 := by
  sorry

end number_comparison_l3237_323766


namespace number_of_laborers_l3237_323719

/-- Proves that the number of laborers is 24 given the salary information --/
theorem number_of_laborers (total_avg : ℝ) (num_supervisors : ℕ) (supervisor_avg : ℝ) (laborer_avg : ℝ) :
  total_avg = 1250 →
  num_supervisors = 6 →
  supervisor_avg = 2450 →
  laborer_avg = 950 →
  ∃ (num_laborers : ℕ), 
    (num_laborers : ℝ) * laborer_avg + (num_supervisors : ℝ) * supervisor_avg = 
    (num_laborers + num_supervisors : ℝ) * total_avg ∧
    num_laborers = 24 :=
by sorry

end number_of_laborers_l3237_323719


namespace count_numbers_theorem_l3237_323704

/-- The count of positive integers less than 50000 with at most three different digits -/
def count_numbers_with_at_most_three_digits : ℕ :=
  let one_digit := 45  -- 5 * 9
  let two_digits_without_zero := 1872  -- 36 * 52
  let two_digits_with_zero := 234  -- 9 * 26
  let three_digits_with_zero := 900  -- 36 * 25
  let three_digits_without_zero := 4452  -- 84 * 53
  one_digit + two_digits_without_zero + two_digits_with_zero + three_digits_with_zero + three_digits_without_zero

/-- The theorem stating that the count of positive integers less than 50000 
    with at most three different digits is 7503 -/
theorem count_numbers_theorem : count_numbers_with_at_most_three_digits = 7503 := by
  sorry

end count_numbers_theorem_l3237_323704


namespace angie_coffee_amount_l3237_323778

/-- Represents the number of cups of coffee brewed per pound of coffee. -/
def cupsPerPound : ℕ := 40

/-- Represents the number of cups of coffee Angie drinks per day. -/
def cupsPerDay : ℕ := 3

/-- Represents the number of days the coffee lasts. -/
def daysLasting : ℕ := 40

/-- Calculates the number of pounds of coffee Angie bought. -/
def coffeeAmount : ℕ := (cupsPerDay * daysLasting) / cupsPerPound

theorem angie_coffee_amount : coffeeAmount = 3 := by
  sorry

end angie_coffee_amount_l3237_323778


namespace triangle_side_length_l3237_323715

-- Define the triangle
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

-- Define right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define 60 degree angle
def SixtyDegreeAngle (A B C : ℝ × ℝ) : Prop :=
  ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2))^2 =
  3 * ((B.1 - A.1)^2 + (B.2 - A.2)^2) * ((C.1 - A.1)^2 + (C.2 - A.2)^2) / 4

-- Define inscribed circle radius
def InscribedCircleRadius (A B C : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), 
    (O.1 - A.1)^2 + (O.2 - A.2)^2 = r^2 ∧
    (O.1 - B.1)^2 + (O.2 - B.2)^2 = r^2 ∧
    (O.1 - C.1)^2 + (O.2 - C.2)^2 = r^2

-- Theorem statement
theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : RightAngle B A C)
  (h3 : SixtyDegreeAngle A B C)
  (h4 : InscribedCircleRadius A B C 8) :
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (24 * Real.sqrt 3 + 24)^2 := by
  sorry

end triangle_side_length_l3237_323715


namespace intersection_sum_l3237_323742

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 5*x + 2
def g (x y : ℝ) : Prop := x + 5*y = 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {p | f p.1 = p.2 ∧ g p.1 p.2}

-- State the theorem
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 :=
  sorry

end intersection_sum_l3237_323742


namespace pet_food_difference_l3237_323729

theorem pet_food_difference (dog_food cat_food : ℕ) 
  (h1 : dog_food = 600) 
  (h2 : cat_food = 327) : 
  dog_food - cat_food = 273 := by
sorry

end pet_food_difference_l3237_323729


namespace impossible_table_fill_l3237_323771

/-- Represents a table filled with natural numbers -/
def Table (n : ℕ) := Fin n → Fin n → ℕ

/-- Checks if a row in the table satisfies the product condition -/
def RowSatisfiesCondition (row : Fin n → ℕ) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ row i * row j = row k

/-- Checks if all elements in the table are distinct and within the range 1 to n^2 -/
def ValidTable (t : Table n) : Prop :=
  (∀ i j, 1 ≤ t i j ∧ t i j ≤ n^2) ∧
  (∀ i₁ j₁ i₂ j₂, (i₁, j₁) ≠ (i₂, j₂) → t i₁ j₁ ≠ t i₂ j₂)

/-- The main theorem stating the impossibility of filling the table -/
theorem impossible_table_fill (n : ℕ) (h : n ≥ 3) :
  ¬∃ (t : Table n), ValidTable t ∧ (∀ i : Fin n, RowSatisfiesCondition (t i)) :=
sorry

end impossible_table_fill_l3237_323771


namespace jack_keeps_half_deer_weight_l3237_323734

/-- Given Jack's hunting habits and the amount of deer he keeps, prove that he keeps half of the total deer weight caught each year. -/
theorem jack_keeps_half_deer_weight 
  (hunts_per_month : ℕ) 
  (hunting_season_months : ℕ) 
  (deers_per_hunt : ℕ) 
  (deer_weight : ℕ) 
  (weight_kept : ℕ) 
  (h1 : hunts_per_month = 6)
  (h2 : hunting_season_months = 3)
  (h3 : deers_per_hunt = 2)
  (h4 : deer_weight = 600)
  (h5 : weight_kept = 10800) : 
  weight_kept / (hunts_per_month * hunting_season_months * deers_per_hunt * deer_weight) = 1 / 2 :=
by sorry

end jack_keeps_half_deer_weight_l3237_323734


namespace luna_budget_l3237_323751

/-- Luna's monthly budget calculation -/
theorem luna_budget (house_rental food phone : ℝ) : 
  food = 0.6 * house_rental →
  phone = 0.1 * food →
  house_rental + food = 240 →
  house_rental + food + phone = 249 := by
  sorry

end luna_budget_l3237_323751


namespace sum_of_max_min_f_l3237_323781

def f (x : ℝ) : ℝ := |x - 2| + |x - 4| - |2*x - 6|

theorem sum_of_max_min_f : 
  ∃ (max min : ℝ), 
    (∀ x, 2 ≤ x ∧ x ≤ 8 → f x ≤ max) ∧ 
    (∃ x, 2 ≤ x ∧ x ≤ 8 ∧ f x = max) ∧
    (∀ x, 2 ≤ x ∧ x ≤ 8 → min ≤ f x) ∧ 
    (∃ x, 2 ≤ x ∧ x ≤ 8 ∧ f x = min) ∧
    max + min = 2 := by
  sorry

end sum_of_max_min_f_l3237_323781


namespace pencils_per_package_l3237_323747

theorem pencils_per_package (pens_per_package : ℕ) (total_pens : ℕ) (pencil_packages : ℕ) :
  pens_per_package = 12 →
  total_pens = 60 →
  total_pens / pens_per_package = pencil_packages →
  total_pens / pencil_packages = 12 :=
by
  sorry

end pencils_per_package_l3237_323747


namespace inscribed_circle_radius_l3237_323760

theorem inscribed_circle_radius 
  (a b c : ℝ) 
  (ha : a = 8) 
  (hb : b = 5) 
  (hc : c = 7) 
  (h_area : (a + b + c) / 2 - 2 = (a + b + c) / 2 * r) : 
  r = 1.8 := by
sorry

end inscribed_circle_radius_l3237_323760


namespace milk_expense_l3237_323752

/-- Proves that the amount spent on milk is 1500, given the total expenses
    excluding milk and savings, the savings amount, and the savings rate. -/
theorem milk_expense (total_expenses_excl_milk : ℕ) (savings : ℕ) (savings_rate : ℚ) :
  total_expenses_excl_milk = 16500 →
  savings = 2000 →
  savings_rate = 1/10 →
  (total_expenses_excl_milk + savings) / (1 - savings_rate) - (total_expenses_excl_milk + savings) = 1500 := by
  sorry

end milk_expense_l3237_323752


namespace base5_product_132_23_l3237_323720

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base 10 number to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Multiplies two base 5 numbers -/
def multiplyBase5 (a b : List Nat) : List Nat :=
  base10ToBase5 (base5ToBase10 a * base5ToBase10 b)

theorem base5_product_132_23 :
  multiplyBase5 [2, 3, 1] [3, 2] = [1, 4, 1, 4] :=
sorry

end base5_product_132_23_l3237_323720


namespace winning_percentage_correct_l3237_323733

/-- Represents the percentage of votes secured by the winning candidate -/
def winning_percentage : ℝ := 70

/-- Represents the total number of valid votes -/
def total_votes : ℕ := 455

/-- Represents the majority of votes by which the winning candidate won -/
def vote_majority : ℕ := 182

/-- Theorem stating that the winning percentage is correct given the conditions -/
theorem winning_percentage_correct : 
  (winning_percentage / 100 * total_votes : ℝ) - 
  ((100 - winning_percentage) / 100 * total_votes : ℝ) = vote_majority := by
  sorry

end winning_percentage_correct_l3237_323733


namespace binary_1011001_to_base5_l3237_323761

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_1011001_to_base5 :
  decimal_to_base5 (binary_to_decimal [true, false, false, true, true, false, true]) = [3, 2, 4] :=
sorry

end binary_1011001_to_base5_l3237_323761


namespace sqrt_two_value_l3237_323701

def f_property (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

theorem sqrt_two_value (f : ℝ → ℝ) (h1 : f_property f) (h2 : f 8 = 3) :
  f (Real.sqrt 2) = 1/2 := by
  sorry

end sqrt_two_value_l3237_323701


namespace crocodile_coloring_l3237_323716

theorem crocodile_coloring (m n : ℕ) (h_m : m > 0) (h_n : n > 0) :
  ∃ f : ℤ × ℤ → Bool,
    ∀ x y : ℤ, f (x, y) ≠ f (x + m, y + n) ∧ f (x, y) ≠ f (x + n, y + m) := by
  sorry

end crocodile_coloring_l3237_323716


namespace zeros_of_continuous_function_l3237_323731

theorem zeros_of_continuous_function (f : ℝ → ℝ) (a b c : ℝ) 
  (h_cont : Continuous f) 
  (h_order : a < b ∧ b < c) 
  (h_sign1 : f a * f b < 0) 
  (h_sign2 : f b * f c < 0) : 
  ∃ (n : ℕ), n ≥ 2 ∧ Even n ∧ 
  (∃ (S : Finset ℝ), S.card = n ∧ (∀ x ∈ S, a < x ∧ x < c ∧ f x = 0)) :=
sorry

end zeros_of_continuous_function_l3237_323731


namespace peters_expression_exists_l3237_323785

/-- An expression type that can represent sums and products of ones -/
inductive Expr
  | one : Expr
  | add : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

/-- Evaluate an expression -/
def eval : Expr → Nat
  | Expr.one => 1
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2

/-- Swap addition and multiplication in an expression -/
def swap : Expr → Expr
  | Expr.one => Expr.one
  | Expr.add e1 e2 => Expr.mul (swap e1) (swap e2)
  | Expr.mul e1 e2 => Expr.add (swap e1) (swap e2)

/-- There exists an expression that evaluates to 2014 and still evaluates to 2014 after swapping + and × -/
theorem peters_expression_exists : ∃ e : Expr, eval e = 2014 ∧ eval (swap e) = 2014 := by
  sorry

end peters_expression_exists_l3237_323785


namespace function_properties_l3237_323796

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - (a + 1) * log x

theorem function_properties (a : ℝ) :
  (∀ x > 0, Monotone (f a)) → a = 1 ∧
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), ∀ x ∈ Set.Icc 1 (Real.exp 1), f a x₀ = -2 ∧ f a x ≥ f a x₀) → a = Real.exp 1 :=
by sorry

end function_properties_l3237_323796


namespace license_plate_count_l3237_323730

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The number of odd digits -/
def num_odd_digits : ℕ := 5

/-- The total number of license plates -/
def total_plates : ℕ := num_letters ^ 3 * num_even_digits * num_odd_digits * num_even_digits

theorem license_plate_count :
  total_plates = 2197000 := by sorry

end license_plate_count_l3237_323730


namespace ellipse_condition_l3237_323794

/-- An equation represents an ellipse if it's of the form (x^2)/a + (y^2)/b = 1,
    where a and b are positive real numbers and a ≠ b. -/
def IsEllipse (m : ℝ) : Prop :=
  m > 0 ∧ 2*m - 1 > 0 ∧ m ≠ 2*m - 1

/-- If the equation (x^2)/m + (y^2)/(2m-1) = 1 represents an ellipse,
    then m > 1/2 and m ≠ 1. -/
theorem ellipse_condition (m : ℝ) :
  IsEllipse m → m > 1/2 ∧ m ≠ 1 := by
  sorry


end ellipse_condition_l3237_323794


namespace root_in_interval_implies_k_range_l3237_323756

/-- Given a quadratic function f(x) = x^2 + (1-k)x - k, if f has a root in the interval (2, 3), 
    then k is in the open interval (2, 3) -/
theorem root_in_interval_implies_k_range (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + (1-k)*x - k
  (∃ x ∈ Set.Ioo 2 3, f x = 0) → k ∈ Set.Ioo 2 3 := by
  sorry

end root_in_interval_implies_k_range_l3237_323756


namespace prime_squared_plus_17_mod_12_l3237_323755

theorem prime_squared_plus_17_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  (p^2 + 17) % 12 = 6 := by
  sorry

end prime_squared_plus_17_mod_12_l3237_323755


namespace pencil_theorem_l3237_323757

def pencil_problem (anna_pencils : ℕ) (harry_pencils : ℕ) (lost_pencils : ℕ) : Prop :=
  anna_pencils = 50 ∧
  harry_pencils = 2 * anna_pencils ∧
  harry_pencils - lost_pencils = 81

theorem pencil_theorem : 
  ∃ (anna_pencils harry_pencils lost_pencils : ℕ),
    pencil_problem anna_pencils harry_pencils lost_pencils ∧ lost_pencils = 19 :=
by
  sorry

end pencil_theorem_l3237_323757


namespace expression_simplification_l3237_323753

theorem expression_simplification (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h : a + b + c = d) :
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2) = 0 := by
sorry


end expression_simplification_l3237_323753
