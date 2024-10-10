import Mathlib

namespace age_difference_l1527_152702

theorem age_difference (a b c : ℕ) : 
  b = 18 →
  b = 2 * c →
  a + b + c = 47 →
  a = b + 2 :=
by sorry

end age_difference_l1527_152702


namespace irene_age_is_46_l1527_152733

-- Define the ages as natural numbers
def eddie_age : ℕ := 92
def becky_age : ℕ := eddie_age / 4
def irene_age : ℕ := 2 * becky_age

-- Theorem statement
theorem irene_age_is_46 : irene_age = 46 := by
  sorry

end irene_age_is_46_l1527_152733


namespace square_circles_l1527_152750

/-- A square in a plane -/
structure Square where
  vertices : Finset (ℝ × ℝ)
  is_square : vertices.card = 4

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Function to check if a circle's diameter has endpoints as vertices of the square -/
def is_valid_circle (s : Square) (c : Circle) : Prop :=
  ∃ (v1 v2 : ℝ × ℝ), v1 ∈ s.vertices ∧ v2 ∈ s.vertices ∧
    v1 ≠ v2 ∧
    c.center = ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2) ∧
    c.radius = Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) / 2

/-- The main theorem -/
theorem square_circles (s : Square) :
  ∃! (circles : Finset Circle), circles.card = 2 ∧
    ∀ c ∈ circles, is_valid_circle s c ∧
    ∀ c, is_valid_circle s c → c ∈ circles :=
  sorry


end square_circles_l1527_152750


namespace rth_term_is_8r_l1527_152711

-- Define the sum of n terms for the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^2 + 1

-- Define the r-th term of the arithmetic progression
def a (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem stating that the r-th term is equal to 8r
theorem rth_term_is_8r (r : ℕ) : a r = 8 * r := by
  sorry

end rth_term_is_8r_l1527_152711


namespace shirt_sales_revenue_function_l1527_152796

/-- The daily net revenue function for shirt sales -/
def daily_net_revenue (x : ℝ) : ℝ :=
  -x^2 + 110*x - 2400

theorem shirt_sales_revenue_function 
  (wholesale_price : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (price_sensitivity : ℝ) 
  (h1 : wholesale_price = 30)
  (h2 : initial_price = 40)
  (h3 : initial_sales = 40)
  (h4 : price_sensitivity = 1)
  (x : ℝ)
  (h5 : x ≥ 40) :
  daily_net_revenue x = (x - wholesale_price) * (initial_sales - (x - initial_price) * price_sensitivity) :=
by
  sorry

#check shirt_sales_revenue_function

end shirt_sales_revenue_function_l1527_152796


namespace blue_fish_with_spots_l1527_152780

theorem blue_fish_with_spots (total_fish : ℕ) (blue_fish : ℕ) (spotted_blue_fish : ℕ) 
  (h1 : total_fish = 60)
  (h2 : blue_fish = total_fish / 3)
  (h3 : spotted_blue_fish = blue_fish / 2) :
  spotted_blue_fish = 10 := by
  sorry

end blue_fish_with_spots_l1527_152780


namespace min_value_of_even_function_l1527_152751

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a^2 - 1) * x - 3 * a

-- State the theorem
theorem min_value_of_even_function (a : ℝ) :
  (∀ x, f a x = f a (-x)) →  -- f is an even function
  (∀ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) → f a x ∈ Set.range (f a)) →  -- domain of f is [4a+2, a^2+1]
  (∃ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) ∧ f a x = -1) →  -- -1 is in the range of f
  (∀ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) → f a x ≥ -1) →  -- -1 is the minimum value
  (∃ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) ∧ f a x = -1)  -- the minimum value of f(x) is -1
  := by sorry

end min_value_of_even_function_l1527_152751


namespace least_n_divisibility_l1527_152788

theorem least_n_divisibility : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), (2 ≤ k) ∧ (k ≤ n) ∧ ((n - 1)^2 % k = 0)) ∧
  (∃ (k : ℕ), (2 ≤ k) ∧ (k ≤ n) ∧ ((n - 1)^2 % k ≠ 0)) ∧
  (∀ (m : ℕ), (m > 0) ∧ (m < n) → 
    (∀ (k : ℕ), (2 ≤ k) ∧ (k ≤ m) → ((m - 1)^2 % k = 0)) ∨
    (∀ (k : ℕ), (2 ≤ k) ∧ (k ≤ m) → ((m - 1)^2 % k ≠ 0))) ∧
  n = 3 :=
by sorry

end least_n_divisibility_l1527_152788


namespace abc_sum_mod_11_l1527_152731

theorem abc_sum_mod_11 (a b c : Nat) : 
  a < 11 → b < 11 → c < 11 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 11 = 3 →
  (8 * c) % 11 = 5 →
  (a + 3 * b) % 11 = 10 →
  (a + b + c) % 11 = 0 := by
  sorry

end abc_sum_mod_11_l1527_152731


namespace nonagon_diagonals_l1527_152781

/-- A convex nonagon is a 9-sided polygon -/
def ConvexNonagon := Nat

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals (n : ConvexNonagon) : Nat :=
  27

theorem nonagon_diagonals :
  ∀ n : ConvexNonagon, num_diagonals n = 27 := by
  sorry

end nonagon_diagonals_l1527_152781


namespace opposite_direction_speed_l1527_152716

/-- Given two people moving in opposite directions for 45 minutes,
    with one moving at 30 kmph and ending up 60 km apart,
    prove that the speed of the other person is 50 kmph. -/
theorem opposite_direction_speed 
  (riya_speed : ℝ) 
  (time : ℝ) 
  (total_distance : ℝ) 
  (h1 : riya_speed = 30) 
  (h2 : time = 45 / 60) 
  (h3 : total_distance = 60) : 
  ∃ (priya_speed : ℝ), priya_speed = 50 ∧ 
    riya_speed * time + priya_speed * time = total_distance :=
by sorry

end opposite_direction_speed_l1527_152716


namespace divisor_problem_l1527_152713

theorem divisor_problem (a b : ℕ) (divisor : ℕ) : 
  (10 ≤ a ∧ a ≤ 99) →  -- a is a two-digit number
  (a = 10 * (a / 10) + (a % 10)) →  -- a is represented in decimal form
  (divisor > 0) →  -- divisor is positive
  (a % divisor = 0) →  -- a is divisible by divisor
  (∀ x y : ℕ, (10 ≤ x ∧ x ≤ 99) → (x % divisor = 0) → (x / 10) * (x % 10) ≤ (a / 10) * (a % 10)) →  -- greatest possible value of b × a
  ((a / 10) * (a % 10) = 35) →  -- b × a = 35
  divisor = 3 :=
by sorry

end divisor_problem_l1527_152713


namespace largest_n_with_prime_differences_l1527_152779

theorem largest_n_with_prime_differences : ∃ n : ℕ, 
  (n = 10) ∧ 
  (∀ m : ℕ, m > 10 → 
    ∃ p : ℕ, Prime p ∧ 2 < p ∧ p < m ∧ ¬(Prime (m - p))) ∧
  (∀ p : ℕ, Prime p → 2 < p → p < 10 → Prime (10 - p)) :=
sorry

end largest_n_with_prime_differences_l1527_152779


namespace binomial_multiply_three_l1527_152749

theorem binomial_multiply_three : 3 * Nat.choose 9 5 = 378 := by sorry

end binomial_multiply_three_l1527_152749


namespace min_value_of_z_l1527_152782

theorem min_value_of_z (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) : 
  ∃ (z_min : ℝ), z_min = (1 + Real.sqrt 5) / 4 ∧ ∀ z, z = x^2 + y^2 → z ≥ z_min :=
by sorry

end min_value_of_z_l1527_152782


namespace science_club_enrollment_l1527_152786

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) :
  total = 150 →
  math = 80 →
  physics = 60 →
  both = 20 →
  total - (math + physics - both) = 30 := by
sorry

end science_club_enrollment_l1527_152786


namespace parallel_vectors_x_value_l1527_152709

/-- Two vectors in R² are parallel if their coordinates are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (4, x + 1)
  parallel a b → x = 1 := by
sorry

end parallel_vectors_x_value_l1527_152709


namespace ribbon_gap_theorem_l1527_152775

theorem ribbon_gap_theorem (R : ℝ) (h : R > 0) :
  let original_length := 2 * Real.pi * R
  let new_length := original_length + 1
  let new_radius := R + (new_length / (2 * Real.pi) - R)
  new_radius - R = 1 / (2 * Real.pi) :=
by sorry

end ribbon_gap_theorem_l1527_152775


namespace remainder_2468135792_mod_101_l1527_152744

theorem remainder_2468135792_mod_101 : 2468135792 % 101 = 52 := by
  sorry

end remainder_2468135792_mod_101_l1527_152744


namespace relationship_of_values_l1527_152766

/-- An odd function f defined on ℝ satisfying f(x) + xf'(x) < 0 for x < 0 -/
class OddDecreasingFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  decreasing : ∀ x < 0, f x + x * (deriv f x) < 0

/-- The main theorem stating the relationship between πf(π), (-2)f(-2), and f(1) -/
theorem relationship_of_values (f : ℝ → ℝ) [OddDecreasingFunction f] :
  π * f π > (-2) * f (-2) ∧ (-2) * f (-2) > f 1 := by
  sorry

end relationship_of_values_l1527_152766


namespace expected_rainfall_l1527_152774

/-- The expected value of total rainfall over 7 days given specific weather conditions --/
theorem expected_rainfall (p_sunny p_light p_heavy : ℝ) (r_light r_heavy : ℝ) (days : ℕ) : 
  p_sunny + p_light + p_heavy = 1 →
  p_sunny = 0.3 →
  p_light = 0.4 →
  p_heavy = 0.3 →
  r_light = 3 →
  r_heavy = 6 →
  days = 7 →
  days * (p_sunny * 0 + p_light * r_light + p_heavy * r_heavy) = 21 :=
by sorry

end expected_rainfall_l1527_152774


namespace combined_weight_is_1170_l1527_152714

/-- The weight Tony can lift in "the curl" exercise -/
def curl_weight : ℝ := 90

/-- The weight Tony can lift in "the military press" exercise -/
def military_press_weight : ℝ := 2 * curl_weight

/-- The weight Tony can lift in "the squat" exercise -/
def squat_weight : ℝ := 5 * military_press_weight

/-- The weight Tony can lift in "the bench press" exercise -/
def bench_press_weight : ℝ := 1.5 * military_press_weight

/-- The combined weight Tony can lift in the squat and bench press exercises -/
def combined_weight : ℝ := squat_weight + bench_press_weight

theorem combined_weight_is_1170 : combined_weight = 1170 := by
  sorry

end combined_weight_is_1170_l1527_152714


namespace correct_division_formula_l1527_152728

theorem correct_division_formula : 
  (240 : ℕ) / (13 + 11) = 240 / (13 + 11) := by sorry

end correct_division_formula_l1527_152728


namespace zoo_animals_count_l1527_152715

theorem zoo_animals_count (penguins : ℕ) (polar_bears : ℕ) : 
  penguins = 21 → polar_bears = 2 * penguins → penguins + polar_bears = 63 := by
  sorry

end zoo_animals_count_l1527_152715


namespace intersection_area_is_three_sqrt_three_half_l1527_152783

/-- Regular tetrahedron with edge length 6 -/
structure RegularTetrahedron where
  edgeLength : ℝ
  edgeLength_eq : edgeLength = 6

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Plane passing through three points -/
structure Plane where
  a : Point3D  -- Vertex A
  m : Point3D  -- Midpoint M
  n : Point3D  -- Point N

/-- The area of intersection between a regular tetrahedron and a plane -/
def intersectionArea (t : RegularTetrahedron) (p : Plane) : ℝ := sorry

/-- Theorem stating that the area of intersection is 3√3/2 -/
theorem intersection_area_is_three_sqrt_three_half (t : RegularTetrahedron) (p : Plane) :
  intersectionArea t p = 3 * Real.sqrt 3 / 2 := by sorry

end intersection_area_is_three_sqrt_three_half_l1527_152783


namespace more_I_than_P_l1527_152729

/-- Sum of digits of a natural number -/
def S (n : ℕ) : ℕ := sorry

/-- Property P: all terms in the sequence n, S(n), S(S(n)),... are even -/
def has_property_P (n : ℕ) : Prop := sorry

/-- Property I: all terms in the sequence n, S(n), S(S(n)),... are odd -/
def has_property_I (n : ℕ) : Prop := sorry

/-- Count of numbers with property P in the range 1 to 2017 -/
def count_P : ℕ := sorry

/-- Count of numbers with property I in the range 1 to 2017 -/
def count_I : ℕ := sorry

theorem more_I_than_P : count_I > count_P := by sorry

end more_I_than_P_l1527_152729


namespace max_sum_with_constraints_l1527_152722

theorem max_sum_with_constraints (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  a + b ≤ 22 / 7 := by
  sorry

end max_sum_with_constraints_l1527_152722


namespace green_notebook_cost_l1527_152777

theorem green_notebook_cost (total_cost black_cost pink_cost : ℕ) 
  (h1 : total_cost = 45)
  (h2 : black_cost = 15)
  (h3 : pink_cost = 10) :
  (total_cost - (black_cost + pink_cost)) / 2 = 10 := by
  sorry

end green_notebook_cost_l1527_152777


namespace inradius_less_than_half_side_height_bound_l1527_152710

/-- Triangle ABC with side lengths a, b, c, angles A, B, C, inradius r, circumradius R, and height h_a from vertex A to side BC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  r : ℝ
  R : ℝ
  h_a : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The inradius is less than half of any side length -/
theorem inradius_less_than_half_side (t : Triangle) : 2 * t.r < t.a := by sorry

/-- The height to side a is at most twice the circumradius times the square of the cosine of half angle A -/
theorem height_bound (t : Triangle) : t.h_a ≤ 2 * t.R * (Real.cos (t.A / 2))^2 := by sorry

end inradius_less_than_half_side_height_bound_l1527_152710


namespace xy_value_l1527_152745

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end xy_value_l1527_152745


namespace mall_profit_analysis_l1527_152771

def average_daily_sales : ℝ := 20
def profit_per_shirt : ℝ := 40
def additional_sales_per_yuan : ℝ := 2

def daily_profit (x : ℝ) : ℝ :=
  (profit_per_shirt - x) * (average_daily_sales + additional_sales_per_yuan * x)

theorem mall_profit_analysis :
  ∃ (f : ℝ → ℝ),
    (∀ x, daily_profit x = f x) ∧
    (f x = -2 * x^2 + 60 * x + 800) ∧
    (∃ x_max, ∀ x, f x ≤ f x_max ∧ x_max = 15) ∧
    (∃ x1 x2, x1 ≠ x2 ∧ f x1 = 1200 ∧ f x2 = 1200 ∧ (x1 = 10 ∨ x1 = 20) ∧ (x2 = 10 ∨ x2 = 20)) :=
by sorry

end mall_profit_analysis_l1527_152771


namespace rational_coordinates_solution_l1527_152764

theorem rational_coordinates_solution (x : ℚ) : ∃ y : ℚ, 2 * x^3 + 2 * y^3 - 3 * x^2 - 3 * y^2 + 1 = 0 := by
  -- We claim that y = 1 - x satisfies the equation
  let y := 1 - x
  -- Existential introduction
  use y
  -- The proof goes here
  sorry

end rational_coordinates_solution_l1527_152764


namespace sum_of_three_times_m_and_half_n_square_diff_minus_square_sum_l1527_152784

-- Part 1
theorem sum_of_three_times_m_and_half_n (m n : ℝ) :
  3 * m + (1/2) * n = 3 * m + (1/2) * n := by sorry

-- Part 2
theorem square_diff_minus_square_sum (a b : ℝ) :
  (a - b)^2 - (a + b)^2 = (a - b)^2 - (a + b)^2 := by sorry

end sum_of_three_times_m_and_half_n_square_diff_minus_square_sum_l1527_152784


namespace abs_neg_three_halves_l1527_152787

theorem abs_neg_three_halves : |(-3/2 : ℚ)| = 3/2 := by
  sorry

end abs_neg_three_halves_l1527_152787


namespace letters_problem_l1527_152735

/-- The number of letters Greta's brother received -/
def brothers_letters : ℕ := sorry

/-- The number of letters Greta received -/
def gretas_letters : ℕ := sorry

/-- The number of letters Greta's mother received -/
def mothers_letters : ℕ := sorry

theorem letters_problem :
  (gretas_letters = brothers_letters + 10) ∧
  (mothers_letters = 2 * (gretas_letters + brothers_letters)) ∧
  (brothers_letters + gretas_letters + mothers_letters = 270) →
  brothers_letters = 40 := by
  sorry

end letters_problem_l1527_152735


namespace inequalities_hold_l1527_152723

theorem inequalities_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  ab ≤ 1 ∧ Real.sqrt a + Real.sqrt b ≤ 2 ∧ a^2 + b^2 ≥ 2 := by
  sorry

end inequalities_hold_l1527_152723


namespace geometric_sequence_sum_l1527_152747

/-- A geometric sequence with common ratio q > 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  (4 * (a 2005)^2 - 8 * (a 2005) + 3 = 0) →
  (4 * (a 2006)^2 - 8 * (a 2006) + 3 = 0) →
  a 2007 + a 2008 = 18 := by
  sorry

end geometric_sequence_sum_l1527_152747


namespace composition_equation_solution_l1527_152756

theorem composition_equation_solution (c : ℝ) : 
  let r (x : ℝ) := 5 * x - 8
  let s (x : ℝ) := 4 * x - c
  r (s 3) = 17 → c = 7 := by
sorry

end composition_equation_solution_l1527_152756


namespace prob_at_least_two_different_fruits_l1527_152736

def num_fruits : ℕ := 4
def num_meals : ℕ := 3

def prob_same_fruit_all_day : ℚ := (1 / num_fruits) ^ num_meals * num_fruits

theorem prob_at_least_two_different_fruits :
  1 - prob_same_fruit_all_day = 15/16 := by
  sorry

end prob_at_least_two_different_fruits_l1527_152736


namespace parabola_h_value_l1527_152759

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * (x - p.h)^2 + p.k

theorem parabola_h_value (p : Parabola) :
  p.a < 0 →
  0 < p.h →
  p.h < 6 →
  p.contains 0 4 →
  p.contains 6 5 →
  p.h = 4 := by
  sorry

#check parabola_h_value

end parabola_h_value_l1527_152759


namespace absolute_value_equation_solution_l1527_152726

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2 * x - 4| = x + 3 :=
by
  -- The unique solution is x = 7
  use 7
  -- Proof goes here
  sorry

end absolute_value_equation_solution_l1527_152726


namespace art_to_maths_ratio_is_one_to_one_l1527_152703

/-- Represents the school supplies problem --/
structure SchoolSupplies where
  total_budget : ℕ
  maths_books : ℕ
  maths_book_price : ℕ
  science_books_diff : ℕ
  science_book_price : ℕ
  music_books_cost : ℕ

/-- The ratio of art books to maths books is 1:1 --/
def art_to_maths_ratio (s : SchoolSupplies) : Prop :=
  let total_spent := s.maths_books * s.maths_book_price + 
                     (s.maths_books + s.science_books_diff) * s.science_book_price + 
                     s.maths_books * s.maths_book_price + 
                     s.music_books_cost
  total_spent ≤ s.total_budget ∧ 
  (s.maths_books : ℚ) / s.maths_books = 1

/-- The main theorem stating that the ratio of art books to maths books is 1:1 --/
theorem art_to_maths_ratio_is_one_to_one (s : SchoolSupplies) 
  (h : s = { total_budget := 500,
             maths_books := 4,
             maths_book_price := 20,
             science_books_diff := 6,
             science_book_price := 10,
             music_books_cost := 160 }) : 
  art_to_maths_ratio s := by
  sorry


end art_to_maths_ratio_is_one_to_one_l1527_152703


namespace trapezoid_bases_solutions_l1527_152798

theorem trapezoid_bases_solutions :
  let valid_pair : ℕ × ℕ → Prop := fun (b₁, b₂) =>
    b₁ + b₂ = 60 ∧ 
    b₁ % 9 = 0 ∧ 
    b₂ % 9 = 0 ∧ 
    b₁ > 0 ∧ 
    b₂ > 0 ∧ 
    (60 : ℝ) * (b₁ + b₂) / 2 = 1800
  ∃! (solutions : List (ℕ × ℕ)),
    solutions.length = 3 ∧ 
    ∀ pair, pair ∈ solutions ↔ valid_pair pair :=
by sorry

end trapezoid_bases_solutions_l1527_152798


namespace math_club_challenge_l1527_152720

theorem math_club_challenge : ((7^2 - 5^2) - 2)^3 = 10648 := by
  sorry

end math_club_challenge_l1527_152720


namespace a8_min_value_l1527_152755

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 3 + a 6 = a 4 + 5
  a2_bound : a 2 ≤ 1

/-- The minimum value of the 8th term in the arithmetic sequence is 9 -/
theorem a8_min_value (seq : ArithmeticSequence) : seq.a 8 ≥ 9 := by
  sorry

end a8_min_value_l1527_152755


namespace surface_area_unchanged_l1527_152743

/-- The surface area of a cube with corner cubes removed -/
def surface_area_with_corners_removed (cube_side_length : ℝ) (corner_side_length : ℝ) : ℝ :=
  6 * cube_side_length^2

/-- The theorem stating that the surface area remains unchanged -/
theorem surface_area_unchanged (cube_side_length : ℝ) (corner_side_length : ℝ) 
  (h1 : cube_side_length = 5) 
  (h2 : corner_side_length = 2) : 
  surface_area_with_corners_removed cube_side_length corner_side_length = 150 := by
  sorry

end surface_area_unchanged_l1527_152743


namespace probability_no_3x3_red_l1527_152700

/-- Represents a 4x4 grid where each cell can be colored red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3x3 subgrid starting at (i, j) is all red -/
def has_red_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ (x y : Fin 3), g (i + x) (j + y) = true

/-- A grid is valid if it doesn't contain a 3x3 red square -/
def is_valid_grid (g : Grid) : Prop :=
  ¬ ∃ (i j : Fin 2), has_red_3x3 g i j

/-- The probability of a single cell being red -/
def p_red : ℚ := 1/2

/-- The total number of possible 4x4 grids -/
def total_grids : ℕ := 2^16

/-- The number of valid 4x4 grids (without 3x3 red squares) -/
def valid_grids : ℕ := 65152

theorem probability_no_3x3_red : 
  (valid_grids : ℚ) / total_grids = 509 / 512 :=
sorry

end probability_no_3x3_red_l1527_152700


namespace perpendicular_vectors_l1527_152758

/-- Given vectors a and b in R², find k such that a ⟂ (a + kb) -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (-2, 1)) (h2 : b = (3, 2)) :
  ∃ k : ℝ, k = 5/4 ∧ a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0 :=
by sorry

end perpendicular_vectors_l1527_152758


namespace geometric_sequence_third_term_l1527_152737

/-- Given a geometric sequence of positive integers where the first term is 5 and the fourth term is 500,
    prove that the third term is equal to 5 * 100^(2/3). -/
theorem geometric_sequence_third_term :
  ∀ (seq : ℕ → ℕ),
    (∀ n, seq (n + 1) / seq n = seq 2 / seq 1) →  -- Geometric sequence condition
    seq 1 = 5 →                                   -- First term is 5
    seq 4 = 500 →                                 -- Fourth term is 500
    seq 3 = 5 * 100^(2/3) :=
by
  sorry


end geometric_sequence_third_term_l1527_152737


namespace smallest_non_factor_product_l1527_152789

theorem smallest_non_factor_product (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  48 % a = 0 → 
  48 % b = 0 → 
  48 % (a * b) ≠ 0 → 
  ∀ c d : ℕ, (c ≠ d ∧ c > 0 ∧ d > 0 ∧ 48 % c = 0 ∧ 48 % d = 0 ∧ 48 % (c * d) ≠ 0) → a * b ≤ c * d →
  a * b = 18 :=
by sorry

end smallest_non_factor_product_l1527_152789


namespace composite_numbers_l1527_152799

theorem composite_numbers (N₁ N₂ : ℕ) : 
  N₁ = 2011 * 2012 * 2013 * 2014 + 1 →
  N₂ = 2012 * 2013 * 2014 * 2015 + 1 →
  ¬(Nat.Prime N₁) ∧ ¬(Nat.Prime N₂) :=
by sorry

end composite_numbers_l1527_152799


namespace min_buses_for_field_trip_l1527_152712

/-- Represents the number of passengers that can be transported by a combination of buses. -/
def transport_capacity (small medium large : ℕ) : ℕ :=
  30 * small + 48 * medium + 72 * large

/-- Represents the total number of buses used. -/
def total_buses (small medium large : ℕ) : ℕ :=
  small + medium + large

theorem min_buses_for_field_trip :
  ∃ (small medium large : ℕ),
    small ≤ 10 ∧
    medium ≤ 15 ∧
    large ≤ 5 ∧
    transport_capacity small medium large ≥ 1230 ∧
    total_buses small medium large = 25 ∧
    (∀ (s m l : ℕ),
      s ≤ 10 →
      m ≤ 15 →
      l ≤ 5 →
      transport_capacity s m l ≥ 1230 →
      total_buses s m l ≥ 25) :=
by sorry

end min_buses_for_field_trip_l1527_152712


namespace chess_tournament_players_l1527_152753

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 240) :
  ∃ n : ℕ, n > 0 ∧ 2 * n * (n - 1) = total_games ∧ n = 12 := by
  sorry

end chess_tournament_players_l1527_152753


namespace car_speed_proof_l1527_152778

/-- Proves that a car traveling for two hours with an average speed of 60 km/h
    and a speed of 30 km/h in the second hour must have a speed of 90 km/h in the first hour. -/
theorem car_speed_proof (x : ℝ) :
  (x + 30) / 2 = 60 →
  x = 90 :=
by sorry

end car_speed_proof_l1527_152778


namespace worker_y_fraction_l1527_152739

theorem worker_y_fraction (fx fy : ℝ) : 
  fx + fy = 1 →
  0.005 * fx + 0.008 * fy = 0.0074 →
  fy = 0.8 := by
sorry

end worker_y_fraction_l1527_152739


namespace tangent_line_implies_a_eq_neg_one_l1527_152730

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x + 1/x - a * Real.log x

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem tangent_line_implies_a_eq_neg_one (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    f a x₀ = tangent_line x₀ ∧ 
    (deriv (f a)) x₀ = (deriv tangent_line) x₀) →
  a = -1 :=
sorry

end

end tangent_line_implies_a_eq_neg_one_l1527_152730


namespace repeating_decimal_sum_l1527_152776

theorem repeating_decimal_sum (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are single digits
  (5 : ℚ) / 13 = (c * 10 + d : ℚ) / 99 →  -- 0.cdcdc... = (c*10 + d) / 99
  c + d = 11 := by
sorry

end repeating_decimal_sum_l1527_152776


namespace height_difference_l1527_152734

/-- The height of the CN Tower in meters -/
def cn_tower_height : ℝ := 553

/-- The height of the Space Needle in meters -/
def space_needle_height : ℝ := 184

/-- Theorem stating the difference in height between the CN Tower and the Space Needle -/
theorem height_difference : cn_tower_height - space_needle_height = 369 := by
  sorry

end height_difference_l1527_152734


namespace largest_divisor_of_sequence_l1527_152727

theorem largest_divisor_of_sequence (n : ℕ) : ∃ (k : ℕ), k = 30 ∧ k ∣ (n^5 - n) ∧ ∀ m : ℕ, m > k → ¬(∀ n : ℕ, m ∣ (n^5 - n)) := by
  sorry

end largest_divisor_of_sequence_l1527_152727


namespace ps_length_is_eight_l1527_152748

/-- Triangle PQR with given side lengths and angle bisector PS -/
structure TrianglePQR where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- PS is the angle bisector of ∠PQR -/
  PS_is_angle_bisector : Bool

/-- The theorem stating that PS = 8 in the given triangle -/
theorem ps_length_is_eight (t : TrianglePQR) 
  (h1 : t.PQ = 8)
  (h2 : t.QR = 15)
  (h3 : t.PR = 17)
  (h4 : t.PS_is_angle_bisector = true) :
  ∃ PS : ℝ, PS = 8 ∧ PS > 0 := by
  sorry


end ps_length_is_eight_l1527_152748


namespace paint_one_third_square_l1527_152705

theorem paint_one_third_square (n : ℕ) (k : ℕ) : n = 18 ∧ k = 6 →
  Nat.choose n k = 18564 := by
  sorry

end paint_one_third_square_l1527_152705


namespace divisible_by_thirteen_l1527_152721

theorem divisible_by_thirteen (n : ℕ) (h : n > 0) :
  ∃ m : ℤ, 4^(2*n - 1) + 3^(n + 1) = 13 * m := by
  sorry

end divisible_by_thirteen_l1527_152721


namespace coin_stack_arrangements_l1527_152752

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of possible face arrangements for n coins where no two adjacent coins are face to face -/
def faceArrangements (n : ℕ) : ℕ := n + 1

theorem coin_stack_arrangements :
  let totalCoins : ℕ := 8
  let goldCoins : ℕ := 5
  let silverCoins : ℕ := 3
  (binomial totalCoins goldCoins) * (faceArrangements totalCoins) = 504 := by sorry

end coin_stack_arrangements_l1527_152752


namespace retailer_profit_is_twenty_percent_l1527_152701

/-- Calculates the percentage profit of a retailer given wholesale price, retail price, and discount percentage. -/
def calculate_percentage_profit (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price / 100
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that under the given conditions, the retailer's percentage profit is 20%. -/
theorem retailer_profit_is_twenty_percent :
  calculate_percentage_profit 99 132 10 = 20 := by
  sorry

end retailer_profit_is_twenty_percent_l1527_152701


namespace f_inequality_l1527_152738

/-- A function satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → x₂ ≤ 1 → f x₁ > f x₂) ∧
  (∀ x : ℝ, f (x + 1) = f (-x + 1))

theorem f_inequality (f : ℝ → ℝ) (h : f_conditions f) :
  f 5.5 < f 7.8 ∧ f 7.8 < f (-2) := by
  sorry

end f_inequality_l1527_152738


namespace inequality_holds_iff_a_in_range_l1527_152718

theorem inequality_holds_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, a * x / (x^2 + 4) < 1.5) ↔ -6 < a ∧ a < 6 := by
  sorry

end inequality_holds_iff_a_in_range_l1527_152718


namespace rectangle_area_perimeter_relation_l1527_152757

theorem rectangle_area_perimeter_relation (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := x + 7
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter ∧ length > 0 ∧ width > 0) → x = 1 := by
  sorry

end rectangle_area_perimeter_relation_l1527_152757


namespace quadratic_equation_solution_exists_l1527_152754

theorem quadratic_equation_solution_exists (a b c : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * b * x + c = 0) ∨ 
  (∃ x : ℝ, b * x^2 + 2 * c * x + a = 0) ∨ 
  (∃ x : ℝ, c * x^2 + 2 * a * x + b = 0) := by
  sorry

end quadratic_equation_solution_exists_l1527_152754


namespace abs_negative_2023_l1527_152772

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_negative_2023_l1527_152772


namespace solve_equation_l1527_152732

theorem solve_equation (x : ℝ) (h : 0.12 / x * 2 = 12) : x = 0.02 := by
  sorry

end solve_equation_l1527_152732


namespace quadratic_equation_solution_l1527_152773

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 + 6*x + 8 + (x + 4)*(x + 6) - 10
  ∃ x₁ x₂ : ℝ, x₁ = -4 + Real.sqrt 5 ∧ x₂ = -4 - Real.sqrt 5 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end quadratic_equation_solution_l1527_152773


namespace kanga_lands_on_84_l1527_152769

def jump_sequence (n : ℕ) : ℕ :=
  9 * n

def kanga_position (n : ℕ) (extra_jumps : ℕ) : ℕ :=
  jump_sequence n + 
  if extra_jumps ≤ 2 then 3 * extra_jumps
  else 6 + (extra_jumps - 2)

theorem kanga_lands_on_84 : 
  ∃ (n : ℕ) (extra_jumps : ℕ), 
    kanga_position n extra_jumps = 84 ∧ 
    kanga_position n extra_jumps ≠ 82 ∧
    kanga_position n extra_jumps ≠ 83 ∧
    kanga_position n extra_jumps ≠ 85 ∧
    kanga_position n extra_jumps ≠ 86 :=
by sorry

end kanga_lands_on_84_l1527_152769


namespace managers_salary_l1527_152746

def employee_count : ℕ := 50
def initial_average_salary : ℚ := 2500
def average_increase : ℚ := 150

theorem managers_salary (manager_salary : ℚ) :
  (employee_count * initial_average_salary + manager_salary) / (employee_count + 1) =
  initial_average_salary + average_increase →
  manager_salary = 10150 := by
sorry

end managers_salary_l1527_152746


namespace min_degree_of_g_l1527_152719

/-- Given polynomials f, g, and h satisfying the equation 5f + 6g = h,
    where deg(f) = 10 and deg(h) = 11, the minimum possible degree of g is 11. -/
theorem min_degree_of_g (f g h : Polynomial ℝ)
  (eq : 5 • f + 6 • g = h)
  (deg_f : Polynomial.degree f = 10)
  (deg_h : Polynomial.degree h = 11) :
  Polynomial.degree g ≥ 11 ∧ ∃ (g' : Polynomial ℝ), Polynomial.degree g' = 11 ∧ 5 • f + 6 • g' = h :=
sorry

end min_degree_of_g_l1527_152719


namespace P_sufficient_not_necessary_for_Q_l1527_152761

theorem P_sufficient_not_necessary_for_Q :
  (∀ a : ℝ, a > 1 → (a - 1) * (a + 1) > 0) ∧
  (∃ a : ℝ, (a - 1) * (a + 1) > 0 ∧ ¬(a > 1)) := by
  sorry

end P_sufficient_not_necessary_for_Q_l1527_152761


namespace river_boat_journey_time_l1527_152707

theorem river_boat_journey_time 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2) 
  (h2 : boat_speed = 6) 
  (h3 : distance = 32) : 
  (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed)) = 12 := by
  sorry

end river_boat_journey_time_l1527_152707


namespace recurring_decimal_product_l1527_152763

theorem recurring_decimal_product : 
  (8 : ℚ) / 99 * (4 : ℚ) / 11 = (32 : ℚ) / 1089 := by sorry

end recurring_decimal_product_l1527_152763


namespace intersection_M_N_l1527_152790

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_M_N : M ∩ N = {5, 7, 9} := by sorry

end intersection_M_N_l1527_152790


namespace share_division_l1527_152742

theorem share_division (total : ℕ) (a b c : ℚ) 
  (h_total : total = 427)
  (h_sum : a + b + c = total)
  (h_ratio : 3 * a = 4 * b ∧ 4 * b = 7 * c) : 
  c = 84 := by
  sorry

end share_division_l1527_152742


namespace unique_solution_m_l1527_152793

/-- A quadratic equation ax^2 + bx + c = 0 has exactly one solution if and only if its discriminant is zero -/
axiom quadratic_one_solution (a b c : ℝ) : 
  (∃! x, a * x^2 + b * x + c = 0) ↔ b^2 - 4*a*c = 0

/-- The value of m for which 3x^2 - 7x + m = 0 has exactly one solution -/
theorem unique_solution_m : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) ↔ m = 49/12 := by sorry

end unique_solution_m_l1527_152793


namespace one_fifths_in_ten_thirds_l1527_152704

theorem one_fifths_in_ten_thirds :
  (10 : ℚ) / 3 / (1 / 5) = 50 / 3 := by sorry

end one_fifths_in_ten_thirds_l1527_152704


namespace nina_money_theorem_l1527_152768

theorem nina_money_theorem (x : ℝ) (h1 : 10 * x = 14 * (x - 3)) : 10 * x = 105 := by
  sorry

end nina_money_theorem_l1527_152768


namespace shielas_paint_colors_l1527_152717

theorem shielas_paint_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (h1 : total_blocks = 196) (h2 : blocks_per_color = 14) : 
  total_blocks / blocks_per_color = 14 := by
  sorry

end shielas_paint_colors_l1527_152717


namespace division_problem_l1527_152725

theorem division_problem (dividend quotient remainder : ℕ) (divisor : ℕ) :
  dividend = 760 →
  quotient = 21 →
  remainder = 4 →
  dividend = divisor * quotient + remainder →
  divisor = 36 := by
sorry

end division_problem_l1527_152725


namespace unknown_number_proof_l1527_152792

theorem unknown_number_proof (x : ℝ) : 
  (0.15 * 25 + 0.12 * x = 9.15) → x = 45 :=
by sorry

end unknown_number_proof_l1527_152792


namespace hunting_company_composition_l1527_152795

theorem hunting_company_composition :
  ∃ (foxes wolves bears : ℕ),
    foxes + wolves + bears = 45 ∧
    59 * foxes + 41 * wolves + 40 * bears = 2008 ∧
    foxes = 10 ∧ wolves = 18 ∧ bears = 17 := by
  sorry

end hunting_company_composition_l1527_152795


namespace potatoes_for_dinner_l1527_152794

def potatoes_for_lunch : ℕ := 5
def total_potatoes : ℕ := 7

theorem potatoes_for_dinner : total_potatoes - potatoes_for_lunch = 2 := by
  sorry

end potatoes_for_dinner_l1527_152794


namespace vector_difference_magnitude_l1527_152791

def unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  unit_vector a → unit_vector b → (a.1 * b.1 + a.2 * b.2 = -1/2) →
  (a.1 - 3*b.1)^2 + (a.2 - 3*b.2)^2 = 13 :=
by sorry

end vector_difference_magnitude_l1527_152791


namespace amy_money_left_l1527_152762

/-- Calculates the amount of money Amy had when she left the fair. -/
def money_left (initial_amount spent : ℕ) : ℕ :=
  initial_amount - spent

/-- Proves that Amy had $11 when she left the fair. -/
theorem amy_money_left :
  money_left 15 4 = 11 := by
  sorry

end amy_money_left_l1527_152762


namespace max_t_value_l1527_152785

theorem max_t_value (f : ℝ → ℝ) (a : ℝ) (t : ℝ) : 
  (∀ x : ℝ, f x = (x + 1)^2) →
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ t → f (x + a) ≤ 2*x - 4) →
  t ≤ 4 :=
by sorry

end max_t_value_l1527_152785


namespace parallelogram_area_12_48_l1527_152765

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 cm and height 48 cm is 576 square centimeters -/
theorem parallelogram_area_12_48 : parallelogram_area 12 48 = 576 := by
  sorry

end parallelogram_area_12_48_l1527_152765


namespace height_matching_problem_sixteenth_answer_l1527_152708

/-- Represents a group of people with their height matches -/
structure HeightGroup :=
  (total : ℕ)
  (one_match : ℕ)
  (two_matches : ℕ)
  (three_matches : ℕ)
  (h_total : total = 16)
  (h_one : one_match = 6)
  (h_two : two_matches = 6)
  (h_three : three_matches = 3)

/-- The number of people accounted for by each match type -/
def accounted_for (g : HeightGroup) : ℕ :=
  g.one_match * 2 + g.two_matches * 3 + g.three_matches * 4

theorem height_matching_problem (g : HeightGroup) :
  accounted_for g = g.total + 3 :=
sorry

theorem sixteenth_answer (g : HeightGroup) :
  g.total - (g.one_match + g.two_matches + g.three_matches) = 1 ∧
  accounted_for g = g.total + 3 →
  3 = g.total - (accounted_for g - 3) :=
sorry

end height_matching_problem_sixteenth_answer_l1527_152708


namespace wallet_value_l1527_152760

def total_bills : ℕ := 12
def five_dollar_bills : ℕ := 4
def five_dollar_value : ℕ := 5
def ten_dollar_value : ℕ := 10

theorem wallet_value :
  (five_dollar_bills * five_dollar_value) +
  ((total_bills - five_dollar_bills) * ten_dollar_value) = 100 :=
by sorry

end wallet_value_l1527_152760


namespace sales_problem_l1527_152767

-- Define the sales revenue function
def sales_revenue (x : ℝ) : ℝ := 1000 * x

-- Define the sales cost function
def sales_cost (x : ℝ) : ℝ := 500 * x + 2000

-- State the theorem
theorem sales_problem :
  -- Condition 1: When x = 0, sales cost is 2000
  sales_cost 0 = 2000 ∧
  -- Condition 2: When x = 2, sales revenue is 2000 and sales cost is 3000
  sales_revenue 2 = 2000 ∧ sales_cost 2 = 3000 ∧
  -- Condition 3: Sales revenue is directly proportional to x (already satisfied by definition)
  -- Condition 4: Sales cost is a linear function of x (already satisfied by definition)
  -- Proof goals:
  -- 1. The functions satisfy all conditions (implicitly proved by the above)
  -- 2. Sales revenue equals sales cost at 4 tons
  (∃ x : ℝ, x = 4 ∧ sales_revenue x = sales_cost x) ∧
  -- 3. Profit at 10 tons is 3000 yuan
  sales_revenue 10 - sales_cost 10 = 3000 :=
by sorry

end sales_problem_l1527_152767


namespace equilateral_triangle_perimeter_area_ratio_l1527_152770

/-- The ratio of the perimeter to the area of an equilateral triangle with side length 8 units is √3/2 -/
theorem equilateral_triangle_perimeter_area_ratio :
  let s : ℝ := 8
  let perimeter : ℝ := 3 * s
  let area : ℝ := s^2 * Real.sqrt 3 / 4
  perimeter / area = Real.sqrt 3 / 2 := by sorry

end equilateral_triangle_perimeter_area_ratio_l1527_152770


namespace end_of_week_stock_l1527_152797

def pencils_per_day : ℕ := 100
def working_days_per_week : ℕ := 5
def initial_stock : ℕ := 80
def pencils_sold : ℕ := 350

theorem end_of_week_stock : 
  pencils_per_day * working_days_per_week + initial_stock - pencils_sold = 230 := by
  sorry

end end_of_week_stock_l1527_152797


namespace two_from_four_combinations_l1527_152706

theorem two_from_four_combinations : Nat.choose 4 2 = 6 := by
  sorry

end two_from_four_combinations_l1527_152706


namespace jacobs_graham_crackers_l1527_152724

/-- Represents the number of graham crackers needed for one s'more -/
def graham_crackers_per_smore : ℕ := 2

/-- Represents the number of marshmallows needed for one s'more -/
def marshmallows_per_smore : ℕ := 1

/-- Represents the number of marshmallows Jacob currently has -/
def current_marshmallows : ℕ := 6

/-- Represents the number of additional marshmallows Jacob needs to buy -/
def additional_marshmallows : ℕ := 18

/-- Theorem stating the number of graham crackers Jacob has -/
theorem jacobs_graham_crackers :
  (current_marshmallows + additional_marshmallows) * graham_crackers_per_smore = 48 := by
  sorry

end jacobs_graham_crackers_l1527_152724


namespace part_I_part_II_l1527_152740

-- Define the statements p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part I
theorem part_I (m : ℝ) (h1 : m > 0) (h2 : ∀ x, p x → q m x) : m ≥ 4 := by
  sorry

-- Part II
theorem part_II (x : ℝ) (h1 : ∀ x, p x ∨ q 5 x) (h2 : ¬∀ x, p x ∧ q 5 x) :
  x ∈ Set.Icc (-3 : ℝ) (-2) ∪ Set.Ioc 6 7 := by
  sorry

end part_I_part_II_l1527_152740


namespace age_determination_l1527_152741

def binary_sum (n : ℕ) : Prop :=
  ∃ (a b c d : Bool),
    n = (if a then 1 else 0) + 
        (if b then 2 else 0) + 
        (if c then 4 else 0) + 
        (if d then 8 else 0)

theorem age_determination (n : ℕ) (h : n < 16) : binary_sum n := by
  sorry

end age_determination_l1527_152741
