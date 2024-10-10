import Mathlib

namespace x_positive_sufficient_not_necessary_for_x_nonzero_l626_62654

theorem x_positive_sufficient_not_necessary_for_x_nonzero :
  (∃ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) :=
by
  sorry

end x_positive_sufficient_not_necessary_for_x_nonzero_l626_62654


namespace cube_difference_l626_62639

/-- Calculates the number of cubes needed for a hollow block -/
def hollow_block_cubes (length width depth : ℕ) : ℕ :=
  2 * length * width + 4 * (length + width) * (depth - 2) - 8 * (depth - 2)

/-- Calculates the number of cubes in a solid block -/
def solid_block_cubes (length width depth : ℕ) : ℕ :=
  length * width * depth

theorem cube_difference (length width depth : ℕ) 
  (h1 : length = 7)
  (h2 : width = 7)
  (h3 : depth = 6) : 
  solid_block_cubes length width depth - hollow_block_cubes length width depth = 100 := by
  sorry

end cube_difference_l626_62639


namespace sum_of_solutions_is_nine_l626_62689

theorem sum_of_solutions_is_nine : 
  let f (x : ℝ) := (12 * x) / (x^2 - 4) - (3 * x) / (x + 2) + 9 / (x - 2)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 9 := by
  sorry

end sum_of_solutions_is_nine_l626_62689


namespace no_rational_roots_l626_62628

theorem no_rational_roots : 
  ∀ (p q : ℤ), q ≠ 0 → 3 * (p / q)^4 - 4 * (p / q)^3 - 9 * (p / q)^2 + 10 * (p / q) + 5 ≠ 0 := by
  sorry

end no_rational_roots_l626_62628


namespace polygon_sides_from_angle_sum_l626_62680

theorem polygon_sides_from_angle_sum :
  ∀ n : ℕ,
  (n - 2) * 180 = 720 →
  n = 6 :=
by
  sorry

end polygon_sides_from_angle_sum_l626_62680


namespace fraction_subtraction_l626_62672

theorem fraction_subtraction : 
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12 := by
  sorry

end fraction_subtraction_l626_62672


namespace reverse_square_digits_l626_62684

theorem reverse_square_digits : ∃! n : ℕ, n > 0 ∧
  (n^2 % 100 = 10 * ((n+1)^2 % 10) + ((n+1)^2 / 10 % 10)) ∧
  ((n+1)^2 % 100 = 10 * (n^2 % 10) + (n^2 / 10 % 10)) :=
sorry

end reverse_square_digits_l626_62684


namespace perpendicular_line_equation_l626_62655

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation 
  (l1 l2 l : Line2D) (P : Point2D) : 
  l1 = Line2D.mk 1 2 (-11) →
  l2 = Line2D.mk 2 1 (-10) →
  pointOnLine P l1 →
  pointOnLine P l2 →
  pointOnLine P l →
  perpendicularLines l l2 →
  l = Line2D.mk 1 (-2) 5 := by
  sorry

end perpendicular_line_equation_l626_62655


namespace value_of_e_l626_62648

theorem value_of_e : (14 : ℕ)^2 * 5^3 * 568 = 13916000 := by
  sorry

end value_of_e_l626_62648


namespace imaginary_part_of_z_l626_62605

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = Complex.abs (1 - Complex.I)) :
  z.im = -Real.sqrt 2 / 2 := by
  sorry

end imaginary_part_of_z_l626_62605


namespace number_equality_l626_62647

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (49/216) * (1/x)) : x = 24.5 := by
  sorry

end number_equality_l626_62647


namespace bahs_equal_to_yahs_l626_62683

/-- Conversion rate from bahs to rahs -/
def bah_to_rah : ℚ := 16 / 10

/-- Conversion rate from rahs to yahs -/
def rah_to_yah : ℚ := 15 / 9

/-- The number of yahs we want to convert -/
def target_yahs : ℚ := 1500

theorem bahs_equal_to_yahs : 
  (target_yahs / (bah_to_rah * rah_to_yah) : ℚ) = 562.5 := by sorry

end bahs_equal_to_yahs_l626_62683


namespace larry_wins_prob_l626_62679

/-- The probability of Larry winning the game --/
def larry_wins : ℚ :=
  let larry_prob : ℚ := 3/4  -- Larry's probability of knocking the bottle off
  let julius_prob : ℚ := 1/4 -- Julius's probability of knocking the bottle off
  let max_rounds : ℕ := 5    -- Maximum number of rounds

  -- Probability of Larry winning in the 1st round
  let p1 : ℚ := larry_prob

  -- Probability of Larry winning in the 3rd round
  let p3 : ℚ := (1 - larry_prob) * julius_prob * larry_prob

  -- Probability of Larry winning in the 5th round
  let p5 : ℚ := ((1 - larry_prob) * julius_prob)^2 * larry_prob

  -- Total probability of Larry winning
  p1 + p3 + p5

/-- Theorem stating that the probability of Larry winning is 825/1024 --/
theorem larry_wins_prob : larry_wins = 825/1024 := by
  sorry

end larry_wins_prob_l626_62679


namespace exists_k_no_roots_l626_62686

/-- A homogeneous real polynomial of degree 2 -/
def HomogeneousPolynomial2 (a b c : ℝ) (x y : ℝ) : ℝ :=
  a * x^2 + b * x * y + c * y^2

/-- A homogeneous real polynomial of degree 3 -/
noncomputable def HomogeneousPolynomial3 (x y : ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem exists_k_no_roots
  (a b c : ℝ)
  (h_pos : b^2 < 4*a*c) :
  ∃ k : ℝ, k > 0 ∧
    ∀ x y : ℝ, x^2 + y^2 < k →
      HomogeneousPolynomial2 a b c x y = HomogeneousPolynomial3 x y →
        x = 0 ∧ y = 0 :=
by sorry

end exists_k_no_roots_l626_62686


namespace race_distance_l626_62619

/-- Given two runners in a race, prove the total distance of the race -/
theorem race_distance (time_A time_B : ℝ) (lead : ℝ) (h1 : time_A = 30) (h2 : time_B = 45) (h3 : lead = 33.333333333333336) :
  ∃ (distance : ℝ), distance = 100 ∧ distance / time_A - distance / time_B = lead / time_A := by
  sorry

end race_distance_l626_62619


namespace midpoint_complex_numbers_l626_62660

theorem midpoint_complex_numbers : 
  let a : ℂ := (1 : ℂ) / (1 + Complex.I)
  let b : ℂ := (1 : ℂ) / (1 - Complex.I)
  let c : ℂ := (a + b) / 2
  c = (1 : ℂ) / 2 := by sorry

end midpoint_complex_numbers_l626_62660


namespace trig_identity_l626_62622

theorem trig_identity (θ : Real) (h : Real.tan (θ + π/4) = 2) :
  Real.sin θ^2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ^2 = -7/5 := by
  sorry

end trig_identity_l626_62622


namespace expected_mass_with_error_l626_62658

/-- The expected mass of 100 metal disks with manufacturing errors -/
theorem expected_mass_with_error (
  nominal_diameter : ℝ)
  (perfect_disk_mass : ℝ)
  (radius_std_dev : ℝ)
  (disk_count : ℕ)
  (h1 : nominal_diameter = 1)
  (h2 : perfect_disk_mass = 100)
  (h3 : radius_std_dev = 0.01)
  (h4 : disk_count = 100) :
  ∃ (expected_mass : ℝ), 
    expected_mass = disk_count * perfect_disk_mass * (1 + 4 * (radius_std_dev / nominal_diameter)^2) ∧
    expected_mass = 10004 :=
by sorry

end expected_mass_with_error_l626_62658


namespace sum_of_squares_parity_l626_62601

theorem sum_of_squares_parity (a b c : ℤ) (h : Odd (a + b + c)) :
  Odd (a^2 + b^2 - c^2 + 2*a*b) := by
  sorry

end sum_of_squares_parity_l626_62601


namespace binomial_expansion_constant_term_l626_62623

theorem binomial_expansion_constant_term (a b : ℝ) (n : ℕ) :
  (2 : ℝ) ^ n = 4 →
  n = 2 →
  (a + b) ^ n = a ^ 2 + 2 * a * b + 9 :=
by sorry

end binomial_expansion_constant_term_l626_62623


namespace cookies_for_students_minimum_recipes_needed_l626_62621

/-- Calculates the minimum number of full recipes needed to provide cookies for students -/
theorem cookies_for_students (original_students : ℕ) (increase_percent : ℕ) 
  (cookies_per_student : ℕ) (cookies_per_recipe : ℕ) : ℕ :=
  let new_students := original_students * (100 + increase_percent) / 100
  let total_cookies := new_students * cookies_per_student
  let recipes_needed := (total_cookies + cookies_per_recipe - 1) / cookies_per_recipe
  recipes_needed

/-- The minimum number of full recipes needed for the given conditions is 33 -/
theorem minimum_recipes_needed : 
  cookies_for_students 108 50 3 15 = 33 := by
  sorry

end cookies_for_students_minimum_recipes_needed_l626_62621


namespace ellipse_and_circle_tangent_lines_l626_62642

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with center (x, y) and radius r -/
structure Circle where
  x : ℝ
  y : ℝ
  r : ℝ
  h_pos : 0 < r

theorem ellipse_and_circle_tangent_lines 
  (C : Ellipse) 
  (E : Circle)
  (h_minor : C.b^2 = 3)
  (h_focus : C.a^2 - C.b^2 = 3)
  (h_radius : E.r^2 = 2)
  (h_center : E.x^2 / C.a^2 + E.y^2 / C.b^2 = 1)
  (k₁ k₂ : ℝ)
  (h_tangent₁ : (E.x - x)^2 + (k₁ * x - E.y)^2 = E.r^2)
  (h_tangent₂ : (E.x - x)^2 + (k₂ * x - E.y)^2 = E.r^2) :
  (C.a^2 = 6 ∧ C.b^2 = 3) ∧ k₁ * k₂ = -1/2 := by sorry

end ellipse_and_circle_tangent_lines_l626_62642


namespace given_curve_is_circle_l626_62652

-- Define a polar coordinate
def PolarCoordinate := ℝ × ℝ

-- Define a circle in terms of its radius
def Circle (radius : ℝ) := {p : PolarCoordinate | p.2 = radius}

-- Define the curve given by the equation r = 5
def GivenCurve := {p : PolarCoordinate | p.2 = 5}

-- Theorem statement
theorem given_curve_is_circle : GivenCurve = Circle 5 := by
  sorry

end given_curve_is_circle_l626_62652


namespace binomial_8_choose_5_l626_62692

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by sorry

end binomial_8_choose_5_l626_62692


namespace amount_saved_christine_savings_l626_62693

/-- Calculates the amount saved by a salesperson given their commission rate, total sales, and allocation for personal needs. -/
theorem amount_saved 
  (commission_rate : ℚ) 
  (total_sales : ℚ) 
  (personal_needs_allocation : ℚ) : ℚ :=
  let commission_earned := commission_rate * total_sales
  let amount_for_personal_needs := personal_needs_allocation * commission_earned
  commission_earned - amount_for_personal_needs

/-- Proves that given the specific conditions, Christine saved $1152. -/
theorem christine_savings : 
  amount_saved (12/100) 24000 (60/100) = 1152 := by
  sorry

end amount_saved_christine_savings_l626_62693


namespace cubic_monotone_implies_a_bound_l626_62629

/-- A function f is monotonically increasing on an interval (a, b) if for any x, y in (a, b) with x < y, we have f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

/-- The cubic function f(x) = ax³ - x² + x - 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

theorem cubic_monotone_implies_a_bound :
  ∀ a : ℝ, MonotonicallyIncreasing (f a) 1 2 → a > 1/3 := by
  sorry

end cubic_monotone_implies_a_bound_l626_62629


namespace equation_equality_relationship_l626_62649

-- Define what an equality is
def IsEquality (s : String) : Prop := true  -- All mathematical statements of the form a = b are equalities

-- Define what an equation is
def IsEquation (s : String) : Prop := IsEquality s ∧ ∃ x, s.contains x  -- An equation is an equality that contains unknowns

-- The statement we want to prove false
def statement : Prop :=
  (∀ s, IsEquation s → IsEquality s) ∧ (∀ s, IsEquality s → IsEquation s)

-- Theorem: The statement is false
theorem equation_equality_relationship : ¬statement := by
  sorry

end equation_equality_relationship_l626_62649


namespace range_of_m_l626_62657

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the property of having two roots in [0, 2]
def has_two_roots_in_interval (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 ∧ 
  f x₁ + m = 0 ∧ f x₂ + m = 0

-- Theorem statement
theorem range_of_m (m : ℝ) :
  has_two_roots_in_interval m → 0 ≤ m ∧ m < 2 :=
by sorry

end range_of_m_l626_62657


namespace linear_function_value_at_negative_two_l626_62687

/-- A linear function passing through a given point -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x

theorem linear_function_value_at_negative_two 
  (k : ℝ) 
  (h : linear_function k 2 = 4) : 
  linear_function k (-2) = -4 := by
sorry

end linear_function_value_at_negative_two_l626_62687


namespace f_equals_g_l626_62671

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (m : ℝ) : ℝ := m^2 - 2*m - 1

theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end f_equals_g_l626_62671


namespace quadratic_discriminant_nonnegative_l626_62651

theorem quadratic_discriminant_nonnegative (a b : ℝ) :
  (∃ x : ℝ, x^2 + a*x + b ≤ 0) → a^2 - 4*b ≥ 0 := by
  sorry

end quadratic_discriminant_nonnegative_l626_62651


namespace p_sufficient_not_necessary_for_q_l626_62625

def p (x : ℝ) : Prop := x = 1

def q (x : ℝ) : Prop := x^3 - 2*x + 1 = 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_for_q_l626_62625


namespace radical_subtraction_l626_62650

theorem radical_subtraction : (5 / Real.sqrt 2) - Real.sqrt (1 / 2) = 2 * Real.sqrt 2 := by
  sorry

end radical_subtraction_l626_62650


namespace not_cube_sum_l626_62638

theorem not_cube_sum (a b : ℤ) : ¬ ∃ (k : ℤ), a^3 + b^3 + 4 = k^3 := by
  sorry

end not_cube_sum_l626_62638


namespace angle_theta_value_l626_62676

theorem angle_theta_value (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (20 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 25 * π / 180 := by
sorry

end angle_theta_value_l626_62676


namespace car_travel_time_l626_62600

/-- Proves that a car traveling at 160 km/h for 800 km takes 5 hours -/
theorem car_travel_time (speed : ℝ) (distance : ℝ) (h1 : speed = 160) (h2 : distance = 800) :
  distance / speed = 5 := by
  sorry

end car_travel_time_l626_62600


namespace mean_of_car_counts_l626_62675

theorem mean_of_car_counts : 
  let counts : List ℕ := [30, 14, 14, 21, 25]
  (counts.sum / counts.length : ℚ) = 20.8 := by sorry

end mean_of_car_counts_l626_62675


namespace triangle_problem_l626_62614

noncomputable def Triangle (a b c : ℝ) (A B C : ℝ) := True

theorem triangle_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c)
  (h_c : c = Real.sqrt 7)
  (h_area : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :
  C = π/3 ∧ a + b = 5 := by
sorry


end triangle_problem_l626_62614


namespace data_transmission_time_data_transmission_problem_l626_62641

theorem data_transmission_time : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (blocks : ℝ) (chunks_per_block : ℝ) (chunks_per_second : ℝ) (time_in_minutes : ℝ) =>
    blocks * chunks_per_block / chunks_per_second / 60 = time_in_minutes

theorem data_transmission_problem :
  data_transmission_time 100 600 150 7 := by
  sorry

end data_transmission_time_data_transmission_problem_l626_62641


namespace factorization_of_quadratic_l626_62674

theorem factorization_of_quadratic (a : ℚ) : 2 * a^2 - 4 * a = 2 * a * (a - 2) := by
  sorry

end factorization_of_quadratic_l626_62674


namespace sum_of_perpendiculars_l626_62643

/-- An equilateral triangle with side length 6 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 6

/-- The centroid of a triangle -/
structure Centroid (T : EquilateralTriangle) where

/-- The perpendicular from the centroid to a side of the triangle -/
def perpendicular (T : EquilateralTriangle) (C : Centroid T) : ℝ := sorry

/-- The theorem stating the sum of perpendiculars from the centroid equals 3√3 -/
theorem sum_of_perpendiculars (T : EquilateralTriangle) (C : Centroid T) :
  3 * (perpendicular T C) = 3 * Real.sqrt 3 := by sorry

end sum_of_perpendiculars_l626_62643


namespace no_solution_iff_m_eq_neg_three_l626_62613

theorem no_solution_iff_m_eq_neg_three (m : ℝ) :
  (∀ x : ℝ, x ≠ -1 → (3 * x) / (x + 1) ≠ m / (x + 1) + 2) ↔ m = -3 := by
  sorry

end no_solution_iff_m_eq_neg_three_l626_62613


namespace integer_properties_l626_62637

theorem integer_properties (m n k : ℕ) (hm : m > 0) (hn : n > 0) : 
  ∃ (a b : ℕ), 
    -- (m+n)^2 + (m-n)^2 is even
    ∃ (c : ℕ), (m + n)^2 + (m - n)^2 = 2 * c ∧
    -- ((m+n)^2 + (m-n)^2) / 2 can be expressed as the sum of squares of two positive integers
    ((m + n)^2 + (m - n)^2) / 2 = a^2 + b^2 ∧
    -- For any integer k, (2k+1)^2 - (2k-1)^2 is divisible by 8
    ∃ (d : ℕ), (2 * k + 1)^2 - (2 * k - 1)^2 = 8 * d :=
by sorry

end integer_properties_l626_62637


namespace alice_has_largest_result_l626_62670

def initial_number : ℕ := 15

def alice_result (n : ℕ) : ℕ := n * 3 - 2 + 4

def bob_result (n : ℕ) : ℕ := n * 2 + 3 - 5

def charlie_result (n : ℕ) : ℕ := ((n + 5) / 2) * 4

theorem alice_has_largest_result :
  alice_result initial_number > bob_result initial_number ∧
  alice_result initial_number > charlie_result initial_number := by
  sorry

end alice_has_largest_result_l626_62670


namespace adult_ticket_price_is_60_l626_62664

/-- Represents the ticket prices and attendance for a football game -/
structure FootballGame where
  adultTicketPrice : ℕ
  childTicketPrice : ℕ
  totalAttendance : ℕ
  adultAttendance : ℕ
  totalRevenue : ℕ

/-- Theorem stating that the adult ticket price is 60 cents -/
theorem adult_ticket_price_is_60 (game : FootballGame) :
  game.childTicketPrice = 25 ∧
  game.totalAttendance = 280 ∧
  game.totalRevenue = 14000 ∧
  game.adultAttendance = 200 →
  game.adultTicketPrice = 60 := by
  sorry

#check adult_ticket_price_is_60

end adult_ticket_price_is_60_l626_62664


namespace circle_a_range_l626_62667

/-- A circle in the xy-plane is represented by the equation (x^2 + y^2 + 2x - 4y + a = 0) -/
def is_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + a = 0

/-- The range of a for which the equation represents a circle -/
theorem circle_a_range :
  {a : ℝ | is_circle a} = Set.Iio 5 :=
by sorry

end circle_a_range_l626_62667


namespace kylie_coins_problem_l626_62610

/-- The number of coins Kylie's father gave her -/
def coins_from_father : ℕ := sorry

theorem kylie_coins_problem : coins_from_father = 8 := by
  have piggy_bank : ℕ := 15
  have from_brother : ℕ := 13
  have given_to_laura : ℕ := 21
  have left_with : ℕ := 15
  
  have total_before_father : ℕ := piggy_bank + from_brother
  have total_after_father : ℕ := total_before_father + coins_from_father
  have after_giving_to_laura : ℕ := total_after_father - given_to_laura
  
  have : after_giving_to_laura = left_with := by sorry
  
  sorry

end kylie_coins_problem_l626_62610


namespace sum_interior_angles_specific_polyhedron_l626_62631

/-- A convex polyhedron with given number of vertices and edges -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- The sum of interior angles of all faces of a convex polyhedron -/
def sum_interior_angles (p : ConvexPolyhedron) : ℝ :=
  sorry

/-- Theorem stating the sum of interior angles for a specific convex polyhedron -/
theorem sum_interior_angles_specific_polyhedron :
  let p : ConvexPolyhedron := ⟨20, 30⟩
  sum_interior_angles p = 6480 :=
by sorry

end sum_interior_angles_specific_polyhedron_l626_62631


namespace gym_class_size_l626_62665

/-- The number of students in the third group -/
def third_group_size : ℕ := 37

/-- The percentage of students in the third group -/
def third_group_percentage : ℚ := 1/2

/-- The total number of students in the gym class -/
def total_students : ℕ := 74

theorem gym_class_size :
  (third_group_size : ℚ) / third_group_percentage = total_students := by
  sorry

end gym_class_size_l626_62665


namespace parallelepiped_volume_l626_62691

/-- Given a rectangular parallelepiped with dimensions l, w, and h, if the shortest distances
    from an interior diagonal to the edges it does not meet are 2√5, 30/√13, and 15/√10,
    then the volume of the parallelepiped is 750. -/
theorem parallelepiped_volume (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) : 
  (l * w / Real.sqrt (l^2 + w^2) = 2 * Real.sqrt 5) →
  (h * w / Real.sqrt (h^2 + w^2) = 30 / Real.sqrt 13) →
  (h * l / Real.sqrt (h^2 + l^2) = 15 / Real.sqrt 10) →
  l * w * h = 750 := by
  sorry

end parallelepiped_volume_l626_62691


namespace cannot_tile_figure_l626_62668

/-- A figure that can be colored such that each 1 × 3 strip covers exactly one colored cell. -/
structure ColoredFigure where
  colored_cells : ℕ

/-- A strip used for tiling. -/
structure Strip where
  width : ℕ
  height : ℕ

/-- Predicate to check if a figure can be tiled with given strips. -/
def CanBeTiled (f : ColoredFigure) (s : Strip) : Prop :=
  f.colored_cells % s.width = 0

theorem cannot_tile_figure (f : ColoredFigure) (s : Strip) 
  (h1 : f.colored_cells = 7)
  (h2 : s.width = 3)
  (h3 : s.height = 1) : 
  ¬CanBeTiled f s := by
  sorry

end cannot_tile_figure_l626_62668


namespace square_cube_remainder_l626_62656

theorem square_cube_remainder (a n : ℕ) 
  (h1 : a^2 % n = 8)
  (h2 : a^3 % n = 25)
  (h3 : n > 25) :
  n = 113 := by
sorry

end square_cube_remainder_l626_62656


namespace twenty_team_tournament_matches_l626_62666

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  num_matches : ℕ

/-- Calculates the number of matches needed in a single-elimination tournament. -/
def matches_needed (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 20 teams requires 19 matches. -/
theorem twenty_team_tournament_matches :
  ∀ t : Tournament, t.num_teams = 20 → t.num_matches = matches_needed t.num_teams := by
  sorry

end twenty_team_tournament_matches_l626_62666


namespace penguin_colony_fish_consumption_l626_62607

theorem penguin_colony_fish_consumption (initial_size : ℕ) : 
  (2 * (2 * initial_size) + 129 = 1077) → 
  (initial_size = 158) := by
  sorry

end penguin_colony_fish_consumption_l626_62607


namespace symmetrical_triangles_are_congruent_l626_62611

/-- Two triangles are symmetrical about a line if each point of one triangle has a corresponding point in the other triangle that is equidistant from the line of symmetry. -/
def symmetrical_triangles (t1 t2 : Set Point) (l : Line) : Prop := sorry

/-- Two triangles are congruent if they have the same shape and size. -/
def congruent_triangles (t1 t2 : Set Point) : Prop := sorry

/-- If two triangles are symmetrical about a line, then they are congruent. -/
theorem symmetrical_triangles_are_congruent (t1 t2 : Set Point) (l : Line) :
  symmetrical_triangles t1 t2 l → congruent_triangles t1 t2 := by sorry

end symmetrical_triangles_are_congruent_l626_62611


namespace regression_line_equation_l626_62602

/-- Given a regression line with slope -1 passing through the point (1, 2),
    prove that its equation is y = -x + 3 -/
theorem regression_line_equation (slope : ℝ) (center : ℝ × ℝ) :
  slope = -1 →
  center = (1, 2) →
  ∀ x y : ℝ, y = slope * (x - center.1) + center.2 ↔ y = -x + 3 :=
by sorry

end regression_line_equation_l626_62602


namespace product_equals_720_l626_62632

theorem product_equals_720 (n : ℕ) (h : n = 5) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) = 720 := by
  sorry

end product_equals_720_l626_62632


namespace total_marbles_l626_62681

/-- Represents the number of marbles of each color -/
structure MarbleCollection where
  yellow : ℕ
  purple : ℕ
  orange : ℕ

/-- The ratio of marbles (yellow:purple:orange) -/
def marble_ratio : MarbleCollection := ⟨2, 4, 6⟩

/-- The number of orange marbles -/
def orange_count : ℕ := 18

/-- Theorem stating the total number of marbles -/
theorem total_marbles (c : MarbleCollection) 
  (h1 : c.yellow * marble_ratio.purple = c.purple * marble_ratio.yellow)
  (h2 : c.yellow * marble_ratio.orange = c.orange * marble_ratio.yellow)
  (h3 : c.orange = orange_count) : 
  c.yellow + c.purple + c.orange = 36 := by
  sorry


end total_marbles_l626_62681


namespace bills_profit_percentage_l626_62627

/-- Represents the original profit percentage -/
def original_profit_percentage : ℝ := 10

/-- Represents the original selling price -/
def original_selling_price : ℝ := 549.9999999999995

/-- Represents the additional profit if the product was bought for 10% less and sold at 30% profit -/
def additional_profit : ℝ := 35

theorem bills_profit_percentage :
  let P := original_selling_price / (1 + original_profit_percentage / 100)
  let new_selling_price := P * 0.9 * 1.3
  new_selling_price = original_selling_price + additional_profit :=
by sorry

end bills_profit_percentage_l626_62627


namespace bowling_team_weight_l626_62697

/-- Given a bowling team with the following properties:
  * 7 original players
  * Original average weight of 103 kg
  * 2 new players join
  * One new player weighs 60 kg
  * New average weight is 99 kg
  Prove that the other new player weighs 110 kg -/
theorem bowling_team_weight (original_players : ℕ) (original_avg : ℝ) 
  (new_players : ℕ) (known_new_weight : ℝ) (new_avg : ℝ) :
  original_players = 7 ∧ 
  original_avg = 103 ∧ 
  new_players = 2 ∧ 
  known_new_weight = 60 ∧ 
  new_avg = 99 →
  ∃ x : ℝ, x = 110 ∧ 
    (original_players * original_avg + known_new_weight + x) / 
    (original_players + new_players) = new_avg :=
by sorry

end bowling_team_weight_l626_62697


namespace polyhedron_property_l626_62633

/-- A convex polyhedron with specific face and vertex properties -/
structure ConvexPolyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  hexagons : ℕ
  vertices : ℕ
  P : ℕ  -- number of pentagons meeting at each vertex
  H : ℕ  -- number of hexagons meeting at each vertex
  T : ℕ  -- number of triangles meeting at each vertex

/-- The properties of the specific polyhedron in the problem -/
def problem_polyhedron : ConvexPolyhedron where
  faces := 38
  triangles := 20
  pentagons := 10
  hexagons := 8
  vertices := 115
  P := 4
  H := 2
  T := 2

/-- The theorem to be proved -/
theorem polyhedron_property (poly : ConvexPolyhedron) 
  (h1 : poly.faces = 38)
  (h2 : poly.triangles = 2 * poly.pentagons)
  (h3 : poly.hexagons = 8)
  (h4 : poly.P = 2 * poly.H)
  (h5 : poly.faces = poly.triangles + poly.pentagons + poly.hexagons)
  (h6 : poly = problem_polyhedron) :
  100 * poly.P + 10 * poly.T + poly.vertices = 535 := by
  sorry

end polyhedron_property_l626_62633


namespace container_capacity_proof_l626_62685

/-- The capacity of a container in liters, given the number of portions and volume per portion in milliliters. -/
def container_capacity (portions : ℕ) (ml_per_portion : ℕ) : ℚ :=
  (portions * ml_per_portion : ℚ) / 1000

/-- Proves that a container with 10 portions of 200 ml each has a capacity of 2 liters. -/
theorem container_capacity_proof :
  container_capacity 10 200 = 2 := by
  sorry

#eval container_capacity 10 200

end container_capacity_proof_l626_62685


namespace stating_max_areas_theorem_l626_62615

/-- Represents a circular disk divided by radii, a secant line, and a non-central chord -/
structure DividedDisk where
  n : ℕ
  radii_count : n > 0

/-- 
Calculates the maximum number of non-overlapping areas in a divided disk.
-/
def max_areas (disk : DividedDisk) : ℕ :=
  4 * disk.n + 1

/-- 
Theorem stating that the maximum number of non-overlapping areas in a divided disk
is equal to 4n + 1, where n is the number of equally spaced radii.
-/
theorem max_areas_theorem (disk : DividedDisk) :
  max_areas disk = 4 * disk.n + 1 := by sorry

end stating_max_areas_theorem_l626_62615


namespace max_value_constraint_l626_62653

theorem max_value_constraint (w x y z : ℝ) (h : 9*w^2 + 4*x^2 + y^2 + 25*z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 201 ∧ 
  (∀ w' x' y' z' : ℝ, 9*w'^2 + 4*x'^2 + y'^2 + 25*z'^2 = 1 → 
    9*w' + 4*x' + 2*y' + 10*z' ≤ max) ∧
  (∃ w'' x'' y'' z'' : ℝ, 9*w''^2 + 4*x''^2 + y''^2 + 25*z''^2 = 1 ∧
    9*w'' + 4*x'' + 2*y'' + 10*z'' = max) := by
  sorry

end max_value_constraint_l626_62653


namespace sphere_volume_ratio_l626_62608

theorem sphere_volume_ratio (r₁ r₂ r₃ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ = 2 * r₁) (h₃ : r₃ = 3 * r₁) :
  (4 / 3) * π * r₃^3 = 3 * ((4 / 3) * π * r₁^3 + (4 / 3) * π * r₂^3) :=
by sorry

end sphere_volume_ratio_l626_62608


namespace unique_cyclic_number_l626_62646

def is_permutation (a b : Nat) : Prop := sorry

def is_six_digit (n : Nat) : Prop := 100000 ≤ n ∧ n < 1000000

theorem unique_cyclic_number : ∃! x : Nat, 
  is_six_digit x ∧ 
  is_six_digit (2*x) ∧ 
  is_six_digit (3*x) ∧ 
  is_six_digit (4*x) ∧ 
  is_six_digit (5*x) ∧ 
  is_six_digit (6*x) ∧
  is_permutation x (2*x) ∧ 
  is_permutation x (3*x) ∧ 
  is_permutation x (4*x) ∧ 
  is_permutation x (5*x) ∧ 
  is_permutation x (6*x) ∧
  x = 142857 :=
by sorry

end unique_cyclic_number_l626_62646


namespace matrix_operation_proof_l626_62659

theorem matrix_operation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -3; 0, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-7, 9; 6, -10]
  2 • A + B = !![1, 3; 6, 0] := by
  sorry

end matrix_operation_proof_l626_62659


namespace notebook_distribution_l626_62688

theorem notebook_distribution (C : ℕ) (N : ℕ) 
  (h1 : N = C^2 / 8)
  (h2 : N = 8 * (C / 2))
  : N = 512 := by
  sorry

end notebook_distribution_l626_62688


namespace ten_cubes_shaded_l626_62661

/-- Represents a 4x4x4 cube with a specific shading pattern -/
structure ShadedCube where
  /-- Total number of smaller cubes -/
  total_cubes : Nat
  /-- Number of cubes per edge -/
  edge_length : Nat
  /-- Number of shaded cubes per face -/
  shaded_per_face : Nat
  /-- Condition: total cubes is 64 -/
  total_is_64 : total_cubes = 64
  /-- Condition: edge length is 4 -/
  edge_is_4 : edge_length = 4
  /-- Condition: 5 cubes are shaded per face -/
  five_shaded : shaded_per_face = 5

/-- The number of uniquely shaded cubes in the ShadedCube -/
def uniquely_shaded_cubes (c : ShadedCube) : Nat :=
  8 + 2  -- 8 corner cubes + 2 center cubes on opposite faces

/-- Theorem stating that exactly 10 cubes are uniquely shaded -/
theorem ten_cubes_shaded (c : ShadedCube) :
  uniquely_shaded_cubes c = 10 := by
  sorry  -- Proof is omitted as per instructions

end ten_cubes_shaded_l626_62661


namespace recycling_problem_l626_62612

/-- Recycling problem -/
theorem recycling_problem (pounds_per_point : ℕ) (gwen_pounds : ℕ) (total_points : ℕ) 
  (h1 : pounds_per_point = 3)
  (h2 : gwen_pounds = 5)
  (h3 : total_points = 6) :
  gwen_pounds / pounds_per_point + (total_points - gwen_pounds / pounds_per_point) * pounds_per_point = 15 :=
by sorry

end recycling_problem_l626_62612


namespace remainder_problem_l626_62663

theorem remainder_problem (k : ℕ) (r : ℕ) (h1 : k > 0) (h2 : k < 38) 
  (h3 : k % 5 = 2) (h4 : k % 6 = 5) (h5 : k % 7 = r) (h6 : r < 7) : k % 11 = 6 := by
  sorry

end remainder_problem_l626_62663


namespace hcf_lcm_problem_l626_62618

theorem hcf_lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 396) (h3 : b = 220) : a = 36 := by
  sorry

end hcf_lcm_problem_l626_62618


namespace subset_implies_m_values_l626_62636

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 2, m^2}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem subset_implies_m_values (m : ℝ) : 
  B m ⊆ A m → m = 0 ∨ m = 2 := by
  sorry

end subset_implies_m_values_l626_62636


namespace michelle_gas_usage_l626_62634

theorem michelle_gas_usage (start_gas end_gas : ℝ) (h1 : start_gas = 0.5) (h2 : end_gas = 0.17) :
  start_gas - end_gas = 0.33 := by
sorry

end michelle_gas_usage_l626_62634


namespace range_of_x_when_proposition_false_l626_62677

theorem range_of_x_when_proposition_false :
  (∀ a : ℝ, -1 ≤ a ∧ a ≤ 3 → ∀ x : ℝ, a * x^2 - (2*a - 1) * x + (3 - a) ≥ 0) →
  ∀ x : ℝ, ((-1 ≤ x ∧ x ≤ 0) ∨ (5/3 ≤ x ∧ x ≤ 4)) :=
by sorry

end range_of_x_when_proposition_false_l626_62677


namespace vector_projection_l626_62690

/-- The projection of vector a in the direction of vector b is equal to √65/5 -/
theorem vector_projection (a b : ℝ × ℝ) : 
  a = (2, 3) → b = (-4, 7) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 65 / 5 := by
  sorry

end vector_projection_l626_62690


namespace tower_block_count_l626_62620

/-- The total number of blocks in a tower after adding more blocks -/
def total_blocks (initial : Float) (added : Float) : Float :=
  initial + added

/-- Theorem: The total number of blocks is the sum of initial and added blocks -/
theorem tower_block_count (initial : Float) (added : Float) :
  total_blocks initial added = initial + added := by
  sorry

end tower_block_count_l626_62620


namespace system_solutions_l626_62678

def is_solution (x y z : ℤ) : Prop :=
  x^2 + y^2 + 2*x + 6*y = -5 ∧
  x^2 + z^2 + 2*x - 4*z = 8 ∧
  y^2 + z^2 + 6*y - 4*z = -3

theorem system_solutions :
  is_solution 1 (-2) (-1) ∧
  is_solution 1 (-2) 5 ∧
  is_solution 1 (-4) (-1) ∧
  is_solution 1 (-4) 5 ∧
  is_solution (-3) (-2) (-1) ∧
  is_solution (-3) (-2) 5 ∧
  is_solution (-3) (-4) (-1) ∧
  is_solution (-3) (-4) 5 :=
by sorry


end system_solutions_l626_62678


namespace power_difference_value_l626_62699

theorem power_difference_value (x m n : ℝ) (hm : x^m = 6) (hn : x^n = 3) : x^(m-n) = 2 := by
  sorry

end power_difference_value_l626_62699


namespace isosceles_right_triangle_hypotenuse_l626_62694

theorem isosceles_right_triangle_hypotenuse (a c : ℝ) :
  a > 0 →  -- Ensure positive side length
  c > 0 →  -- Ensure positive hypotenuse length
  c^2 = 2 * a^2 →  -- Pythagorean theorem for isosceles right triangle
  2 * a + c = 4 + 4 * Real.sqrt 2 →  -- Perimeter condition
  c = 4 := by
sorry

end isosceles_right_triangle_hypotenuse_l626_62694


namespace exactly_three_false_l626_62698

-- Define the type for statements
inductive Statement
| one : Statement
| two : Statement
| three : Statement
| four : Statement

-- Define a function to evaluate the truth value of a statement
def evaluate : Statement → (Statement → Bool) → Bool
| Statement.one, f => (f Statement.one && ¬f Statement.two && ¬f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && f Statement.two && ¬f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && ¬f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && ¬f Statement.two && ¬f Statement.three && f Statement.four)
| Statement.two, f => (¬f Statement.one && f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && f Statement.two && ¬f Statement.three && f Statement.four) ||
                      (f Statement.one && ¬f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (f Statement.one && ¬f Statement.two && ¬f Statement.three && f Statement.four) ||
                      (¬f Statement.one && f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (f Statement.one && f Statement.two && ¬f Statement.three && ¬f Statement.four)
| Statement.three, f => (¬f Statement.one && ¬f Statement.two && f Statement.three && ¬f Statement.four)
| Statement.four, f => (f Statement.one && f Statement.two && f Statement.three && f Statement.four)

-- Theorem statement
theorem exactly_three_false :
  ∃ (f : Statement → Bool),
    (∀ s, evaluate s f = f s) ∧
    (f Statement.one = false ∧
     f Statement.two = false ∧
     f Statement.three = true ∧
     f Statement.four = false) :=
by sorry

end exactly_three_false_l626_62698


namespace complex_number_equivalence_l626_62662

theorem complex_number_equivalence : 
  let z : ℂ := (1 - I) / (2 + I)
  z = 1/5 - 3/5*I :=
by
  sorry

end complex_number_equivalence_l626_62662


namespace modular_inverse_of_5_mod_17_l626_62604

theorem modular_inverse_of_5_mod_17 : 
  ∃! x : ℕ, x ∈ Finset.range 17 ∧ (5 * x) % 17 = 1 :=
by
  sorry

end modular_inverse_of_5_mod_17_l626_62604


namespace angle_between_vectors_l626_62617

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given non-zero vectors a and b such that ‖a + 3b‖ = ‖a - 3b‖, 
    the angle between them is 90 degrees. -/
theorem angle_between_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : ‖a + 3 • b‖ = ‖a - 3 • b‖) : 
    Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 2 := by
  sorry

end angle_between_vectors_l626_62617


namespace dividend_divisor_problem_l626_62669

theorem dividend_divisor_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  y / x = 6 ∧ x + y + 6 = 216 → x = 30 ∧ y = 180 := by
  sorry

end dividend_divisor_problem_l626_62669


namespace quarter_circles_sum_limit_l626_62630

theorem quarter_circles_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |n * (π * D / (4 * n)) - (π * D / 4)| < ε :=
sorry

end quarter_circles_sum_limit_l626_62630


namespace unique_number_with_three_prime_factors_l626_62603

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 7^n + 1 → 
  Odd n → 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) → 
  x = 16808 := by
sorry

end unique_number_with_three_prime_factors_l626_62603


namespace pauls_recycling_bags_l626_62626

theorem pauls_recycling_bags (x : ℕ) : 
  (∃ (bags_on_sunday : ℕ), 
    bags_on_sunday = 3 ∧ 
    (∀ (cans_per_bag : ℕ), cans_per_bag = 8 → 
      (∀ (total_cans : ℕ), total_cans = 72 → 
        cans_per_bag * (x + bags_on_sunday) = total_cans))) → 
  x = 6 := by sorry

end pauls_recycling_bags_l626_62626


namespace base4_calculation_l626_62645

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Multiplication in base 4 --/
def mulBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a * base4ToBase10 b)

/-- Division in base 4 --/
def divBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a / base4ToBase10 b)

theorem base4_calculation : 
  mulBase4 (divBase4 321 3) 21 = 2223 := by sorry

end base4_calculation_l626_62645


namespace thief_speed_calculation_chase_problem_l626_62624

/-- Represents the chase scenario between a policeman and a thief -/
structure ChaseScenario where
  initial_distance : ℝ  -- in meters
  policeman_speed : ℝ   -- in km/hr
  thief_distance : ℝ    -- in meters
  thief_speed : ℝ       -- in km/hr

/-- Theorem stating the relationship between the given parameters and the thief's speed -/
theorem thief_speed_calculation (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 160)
  (h2 : scenario.policeman_speed = 10)
  (h3 : scenario.thief_distance = 640) :
  scenario.thief_speed = 8 := by
  sorry

/-- Main theorem proving the specific case -/
theorem chase_problem : 
  ∃ (scenario : ChaseScenario), 
    scenario.initial_distance = 160 ∧ 
    scenario.policeman_speed = 10 ∧ 
    scenario.thief_distance = 640 ∧ 
    scenario.thief_speed = 8 := by
  sorry

end thief_speed_calculation_chase_problem_l626_62624


namespace equal_savings_l626_62635

theorem equal_savings (your_initial : ℕ) (friend_initial : ℕ) (your_rate : ℕ) (friend_rate : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  friend_initial = 210 →
  your_rate = 7 →
  friend_rate = 5 →
  weeks = 25 →
  your_initial + your_rate * weeks = friend_initial + friend_rate * weeks :=
by sorry

end equal_savings_l626_62635


namespace side_length_S2_is_correct_l626_62695

/-- The side length of square S2 in a specific arrangement of rectangles and squares. -/
def side_length_S2 : ℕ :=
  let total_width : ℕ := 4422
  let total_height : ℕ := 2420
  -- S1 and S3 have the same side length, which is also the smaller dimension of R1 and R2
  -- Let r be this common side length
  -- Let s be the side length of S2
  -- From the height: 2r + s = total_height
  -- From the width: 2r + 3s = total_width
  -- Solving this system of equations gives s = 1001
  1001

/-- Theorem stating that the side length of S2 is correct given the conditions. -/
theorem side_length_S2_is_correct :
  let total_width : ℕ := 4422
  let total_height : ℕ := 2420
  ∃ (r : ℕ),
    (2 * r + side_length_S2 = total_height) ∧
    (2 * r + 3 * side_length_S2 = total_width) :=
by sorry

#eval side_length_S2  -- Should output 1001

end side_length_S2_is_correct_l626_62695


namespace intersection_and_union_when_a_is_negative_four_condition_for_b_subset_complement_a_l626_62682

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

-- Theorem for the first part of the problem
theorem intersection_and_union_when_a_is_negative_four :
  (A ∩ B (-4)) = {x | 1/2 ≤ x ∧ x < 2} ∧
  (A ∪ B (-4)) = {x | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem for the second part of the problem
theorem condition_for_b_subset_complement_a (a : ℝ) :
  (B a ∩ Aᶜ = B a) ↔ a ≥ -1/4 := by sorry

end intersection_and_union_when_a_is_negative_four_condition_for_b_subset_complement_a_l626_62682


namespace tricycle_count_proof_l626_62616

/-- Represents the number of wheels on a vehicle -/
def wheels : Nat → Nat
  | 0 => 2  -- bicycle
  | 1 => 3  -- tricycle
  | 2 => 2  -- scooter
  | _ => 0  -- undefined for other values

/-- Represents the count of each type of vehicle -/
structure VehicleCounts where
  bicycles : Nat
  tricycles : Nat
  scooters : Nat

theorem tricycle_count_proof (counts : VehicleCounts) : 
  counts.bicycles + counts.tricycles + counts.scooters = 10 →
  wheels 0 * counts.bicycles + wheels 1 * counts.tricycles + wheels 2 * counts.scooters = 29 →
  counts.tricycles = 9 := by
  sorry

#check tricycle_count_proof

end tricycle_count_proof_l626_62616


namespace total_dogs_count_l626_62673

/-- Represents the number of dogs in the Smartpup Training Center -/
structure DogCount where
  sit : ℕ
  stay : ℕ
  roll_over : ℕ
  jump : ℕ
  sit_stay : ℕ
  stay_roll : ℕ
  sit_roll : ℕ
  jump_stay : ℕ
  sit_stay_roll : ℕ
  no_tricks : ℕ

/-- Theorem stating that the total number of dogs is 150 given the specified conditions -/
theorem total_dogs_count (d : DogCount) 
  (h1 : d.sit = 60)
  (h2 : d.stay = 40)
  (h3 : d.roll_over = 45)
  (h4 : d.jump = 50)
  (h5 : d.sit_stay = 25)
  (h6 : d.stay_roll = 15)
  (h7 : d.sit_roll = 20)
  (h8 : d.jump_stay = 5)
  (h9 : d.sit_stay_roll = 10)
  (h10 : d.no_tricks = 5) : 
  d.sit + d.stay + d.roll_over + d.jump - d.sit_stay - d.stay_roll - d.sit_roll - 
  d.jump_stay + d.sit_stay_roll + d.no_tricks = 150 := by
  sorry


end total_dogs_count_l626_62673


namespace share_ratio_l626_62696

def problem (total a b c : ℚ) : Prop :=
  total = 527 ∧
  a = 372 ∧
  b = 93 ∧
  c = 62 ∧
  a = (2/3) * b ∧
  total = a + b + c

theorem share_ratio (total a b c : ℚ) (h : problem total a b c) :
  b / c = 3 / 2 := by
  sorry

end share_ratio_l626_62696


namespace equation_solution_l626_62644

theorem equation_solution (x : ℝ) : (x + 2)^(x + 3) = 1 → x = -3 ∨ x = -1 := by
  sorry

end equation_solution_l626_62644


namespace sum_of_zeros_is_14_l626_62640

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the final parabola after transformations
def final_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

-- Define the zeros of the final parabola
def p : ℝ := 8
def q : ℝ := 6

-- Theorem statement
theorem sum_of_zeros_is_14 : p + q = 14 := by sorry

end sum_of_zeros_is_14_l626_62640


namespace three_lines_cannot_form_triangle_l626_62606

/-- Three lines in the plane -/
structure ThreeLines where
  l1 : ℝ → ℝ → Prop
  l2 : ℝ → ℝ → Prop
  l3 : ℝ → ℝ → ℝ → Prop

/-- The condition that three lines cannot form a triangle -/
def cannotFormTriangle (lines : ThreeLines) (m : ℝ) : Prop :=
  (∃ (x y : ℝ), lines.l1 x y ∧ lines.l2 x y ∧ lines.l3 m x y) ∨
  (∃ (a b : ℝ), ∀ (x y : ℝ), (lines.l1 x y ↔ y = a*x + b) ∧ 
                              (lines.l3 m x y ↔ y = a*x + (1 - a*m)/m)) ∨
  (∃ (a b : ℝ), ∀ (x y : ℝ), (lines.l2 x y ↔ y = a*x + b) ∧ 
                              (lines.l3 m x y ↔ y = a*x + (1 + a*m)/m))

/-- The given lines -/
def givenLines : ThreeLines :=
  { l1 := λ x y => 2*x - 3*y + 1 = 0
  , l2 := λ x y => 4*x + 3*y + 5 = 0
  , l3 := λ m x y => m*x - y - 1 = 0 }

theorem three_lines_cannot_form_triangle :
  {m : ℝ | cannotFormTriangle givenLines m} = {-4/3, 2/3, 4/3} := by sorry

end three_lines_cannot_form_triangle_l626_62606


namespace quilt_shaded_fraction_l626_62609

/-- Represents a square quilt made of smaller squares -/
structure Quilt :=
  (size : Nat)
  (shaded_row : Nat)
  (shaded_column : Nat)

/-- Calculates the fraction of shaded area in a quilt -/
def shaded_fraction (q : Quilt) : Rat :=
  let total_squares := q.size * q.size
  let shaded_squares := q.size + q.size - 1
  shaded_squares / total_squares

/-- Theorem stating that for a 4x4 quilt with one shaded row and column, 
    the shaded fraction is 7/16 -/
theorem quilt_shaded_fraction :
  ∀ (q : Quilt), q.size = 4 → shaded_fraction q = 7 / 16 := by
  sorry

end quilt_shaded_fraction_l626_62609
