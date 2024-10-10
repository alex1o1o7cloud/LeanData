import Mathlib

namespace class_size_l418_41874

theorem class_size (total : ℕ) (brown_eyes : ℕ) (brown_eyes_black_hair : ℕ) : 
  (3 * brown_eyes = 2 * total) →
  (2 * brown_eyes_black_hair = brown_eyes) →
  (brown_eyes_black_hair = 6) →
  total = 18 :=
by
  sorry

end class_size_l418_41874


namespace blood_expiration_theorem_l418_41866

-- Define the factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the blood expiration time in seconds
def blood_expiration_time : ℕ := factorial 7

-- Define the donation time
def donation_time : ℕ := 18 * 60 * 60  -- 6 PM in seconds

-- Define the expiration datetime
def expiration_datetime : ℕ := donation_time + blood_expiration_time

-- Theorem to prove
theorem blood_expiration_theorem :
  expiration_datetime = 19 * 60 * 60 + 24 * 60 :=  -- 7:24 PM in seconds
by sorry

end blood_expiration_theorem_l418_41866


namespace power_digits_theorem_l418_41870

/-- The number of digits to the right of the decimal place in a given number -/
def decimalDigits (x : ℝ) : ℕ :=
  sorry

/-- The result of raising a number to a power -/
def powerResult (base : ℝ) (exponent : ℕ) : ℝ :=
  base ^ exponent

theorem power_digits_theorem :
  let base := 10^4 * 3.456789
  decimalDigits (powerResult base 11) = 22 := by
  sorry

end power_digits_theorem_l418_41870


namespace xiaoming_multiplication_l418_41810

theorem xiaoming_multiplication (a : ℝ) : 
  20.18 * a = 20.18 * (a - 1) + 2270.25 → a = 113.5 := by
  sorry

end xiaoming_multiplication_l418_41810


namespace special_parallelogram_sides_l418_41818

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- The perimeter of the parallelogram
  perimeter : ℝ
  -- The measure of the acute angle in radians
  acute_angle : ℝ
  -- The ratio of the parts of the obtuse angle divided by the diagonal
  obtuse_angle_ratio : ℝ
  -- The length of the shorter side
  short_side : ℝ
  -- The length of the longer side
  long_side : ℝ
  -- The perimeter is 90 cm
  perimeter_eq : perimeter = 90
  -- The acute angle is 60 degrees (π/3 radians)
  acute_angle_eq : acute_angle = π / 3
  -- The obtuse angle is divided in a 1:3 ratio
  obtuse_angle_ratio_eq : obtuse_angle_ratio = 1 / 3
  -- The perimeter is the sum of all sides
  perimeter_sum : perimeter = 2 * (short_side + long_side)
  -- The shorter side is half the longer side (derived from the 60° angle)
  side_ratio : short_side = long_side / 2

/-- Theorem: The sides of the special parallelogram are 15 cm and 30 cm -/
theorem special_parallelogram_sides (p : SpecialParallelogram) :
  p.short_side = 15 ∧ p.long_side = 30 := by
  sorry

end special_parallelogram_sides_l418_41818


namespace rebecca_tips_calculation_l418_41855

/-- Rebecca's hair salon earnings calculation -/
def rebeccaEarnings (haircut_price perm_price dye_price dye_cost : ℕ) 
  (num_haircuts num_perms num_dyes : ℕ) (total_end_day : ℕ) : ℕ :=
  let service_earnings := haircut_price * num_haircuts + perm_price * num_perms + dye_price * num_dyes
  let dye_costs := dye_cost * num_dyes
  let tips := total_end_day - (service_earnings - dye_costs)
  tips

/-- Theorem stating that Rebecca's tips are $50 given the problem conditions -/
theorem rebecca_tips_calculation :
  rebeccaEarnings 30 40 60 10 4 1 2 310 = 50 := by
  sorry

end rebecca_tips_calculation_l418_41855


namespace students_taller_than_yoongi_l418_41833

theorem students_taller_than_yoongi 
  (total_students : ℕ) 
  (shorter_than_yoongi : ℕ) 
  (h1 : total_students = 20) 
  (h2 : shorter_than_yoongi = 11) : 
  total_students - shorter_than_yoongi - 1 = 8 := by
  sorry

end students_taller_than_yoongi_l418_41833


namespace trajectory_is_line_segment_l418_41838

/-- The trajectory of a point P, where the sum of its distances to two fixed points A(-1, 0) and B(1, 0) is constant 2, is the line segment AB. -/
theorem trajectory_is_line_segment (P : ℝ × ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let dist (X Y : ℝ × ℝ) := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  (dist P A + dist P B = 2) → 
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (2*t - 1, 0) :=
by sorry

end trajectory_is_line_segment_l418_41838


namespace pizza_cost_per_pizza_l418_41888

theorem pizza_cost_per_pizza (num_pizzas : ℕ) (num_toppings : ℕ) 
  (cost_per_topping : ℚ) (tip : ℚ) (total_cost : ℚ) :
  num_pizzas = 3 →
  num_toppings = 4 →
  cost_per_topping = 1 →
  tip = 5 →
  total_cost = 39 →
  ∃ (cost_per_pizza : ℚ), 
    cost_per_pizza = 10 ∧ 
    num_pizzas * cost_per_pizza + num_toppings * cost_per_topping + tip = total_cost :=
by sorry

end pizza_cost_per_pizza_l418_41888


namespace inscribed_square_side_length_l418_41863

/-- A right triangle with side lengths 6, 8, and 10 -/
structure RightTriangle :=
  (PQ : ℝ) (QR : ℝ) (PR : ℝ)
  (right_angle : PQ^2 + QR^2 = PR^2)
  (PQ_eq : PQ = 6)
  (QR_eq : QR = 8)
  (PR_eq : PR = 10)

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) :=
  (side_length : ℝ)
  (on_hypotenuse : side_length ≤ t.PR)
  (on_leg1 : side_length ≤ t.PQ)
  (on_leg2 : side_length ≤ t.QR)

/-- The side length of the inscribed square is 3 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 3 := by sorry

end inscribed_square_side_length_l418_41863


namespace line_through_point_parallel_to_y_axis_l418_41830

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane -/
structure Line where
  equation : ℝ → Prop

/-- Checks if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.equation p.x

/-- Checks if a line is parallel to the y-axis -/
def Line.parallelToYAxis (l : Line) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, l.equation x ↔ x = k

theorem line_through_point_parallel_to_y_axis 
  (A : Point) 
  (h_A : A.x = -3 ∧ A.y = 1) 
  (l : Line) 
  (h_parallel : l.parallelToYAxis) 
  (h_passes : A.liesOn l) : 
  ∀ x : ℝ, l.equation x ↔ x = -3 :=
sorry

end line_through_point_parallel_to_y_axis_l418_41830


namespace quadratic_equation_shift_l418_41859

theorem quadratic_equation_shift (a h k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3 ∧ 
   ∀ x : ℝ, a * (x - h)^2 + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ y₁ y₂ : ℝ, y₁ = 0 ∧ y₂ = 4 ∧ 
   ∀ y : ℝ, a * (y - h - 1)^2 + k = 0 ↔ y = y₁ ∨ y = y₂) :=
by sorry

end quadratic_equation_shift_l418_41859


namespace inequality_proof_l418_41802

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  1/x + 1/y ≥ 3 + 2*Real.sqrt 2 := by
  sorry

end inequality_proof_l418_41802


namespace probability_king_of_diamonds_l418_41858

/-- Represents a standard playing card suit -/
inductive Suit
| Spades
| Hearts
| Diamonds
| Clubs

/-- Represents a standard playing card rank -/
inductive Rank
| Ace
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight
| Nine
| Ten
| Jack
| Queen
| King

/-- Represents a standard playing card -/
structure Card where
  rank : Rank
  suit : Suit

def standardDeck : Finset Card := sorry

/-- The probability of drawing a specific card from a standard deck -/
def probabilityOfCard (c : Card) : ℚ :=
  1 / (Finset.card standardDeck)

theorem probability_king_of_diamonds :
  probabilityOfCard ⟨Rank.King, Suit.Diamonds⟩ = 1 / 52 := by
  sorry

end probability_king_of_diamonds_l418_41858


namespace boat_current_speed_l418_41854

theorem boat_current_speed 
  (boat_speed : ℝ) 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : upstream_time = 20 / 60)
  (h3 : downstream_time = 15 / 60) :
  ∃ (current_speed : ℝ),
    (boat_speed - current_speed) * upstream_time = 
    (boat_speed + current_speed) * downstream_time ∧ 
    current_speed = 16 / 7 := by
  sorry

end boat_current_speed_l418_41854


namespace cone_radius_is_one_l418_41846

/-- Given a cone whose surface area is 3π and whose lateral surface unfolds into a semicircle,
    prove that the radius of its base is 1. -/
theorem cone_radius_is_one (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  π * l = 2 * π * r →  -- lateral surface unfolds into a semicircle
  π * r^2 + π * r * l = 3 * π →  -- surface area is 3π
  r = 1 := by
  sorry

end cone_radius_is_one_l418_41846


namespace line_inclination_45_implies_a_equals_1_l418_41897

/-- If the line ax + (2a - 3)y = 0 has an angle of inclination of 45°, then a = 1 -/
theorem line_inclination_45_implies_a_equals_1 (a : ℝ) : 
  (∃ x y : ℝ, a * x + (2 * a - 3) * y = 0 ∧ 
   Real.arctan ((3 - 2 * a) / a) = π / 4) → 
  a = 1 := by
  sorry

end line_inclination_45_implies_a_equals_1_l418_41897


namespace oak_trees_in_park_l418_41807

/-- The number of oak trees remaining in a park after some are cut down -/
def remaining_oak_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that 7 oak trees remain after cutting down 2 from an initial 9 -/
theorem oak_trees_in_park : remaining_oak_trees 9 2 = 7 := by
  sorry

end oak_trees_in_park_l418_41807


namespace shaded_to_large_square_ratio_l418_41883

theorem shaded_to_large_square_ratio :
  let large_square_side : ℕ := 5
  let unit_squares_count : ℕ := large_square_side ^ 2
  let half_squares_in_shaded : ℕ := 5
  let shaded_area : ℚ := (half_squares_in_shaded : ℚ) / 2
  let large_square_area : ℕ := unit_squares_count
  (shaded_area : ℚ) / (large_square_area : ℚ) = 1 / 10 := by
  sorry

end shaded_to_large_square_ratio_l418_41883


namespace inequality_proof_l418_41865

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (a^2 + b^2 + c^2) ≥ 9 * a * b * c :=
by sorry

end inequality_proof_l418_41865


namespace five_twelve_thirteen_pythagorean_triple_l418_41848

/-- A Pythagorean triple is a set of three positive integers a, b, and c that satisfy a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean_triple :
  isPythagoreanTriple 5 12 13 := by
  sorry

end five_twelve_thirteen_pythagorean_triple_l418_41848


namespace blankets_per_person_l418_41821

/-- Proves that the number of blankets each person gave on the first day is 2 --/
theorem blankets_per_person (team_size : Nat) (last_day_blankets : Nat) (total_blankets : Nat) :
  team_size = 15 →
  last_day_blankets = 22 →
  total_blankets = 142 →
  ∃ (first_day_blankets : Nat),
    first_day_blankets * team_size + 3 * (first_day_blankets * team_size) + last_day_blankets = total_blankets ∧
    first_day_blankets = 2 := by
  sorry

#check blankets_per_person

end blankets_per_person_l418_41821


namespace triangle_proof_l418_41864

theorem triangle_proof (A B C : Real) (a b c : Real) (D : Real) :
  a = Real.sqrt 19 →
  (Real.sin B + Real.sin C) / (Real.cos B + Real.cos A) = (Real.cos B - Real.cos A) / Real.sin C →
  ∃ (D : Real), D ∈ Set.Icc 0 1 ∧ 
    (3 * (1 - D)) / (4 * D) = 1 ∧
    ((1 - D) * b + D * c) * (c * Real.cos A) = 0 →
  A = 2 * Real.pi / 3 ∧
  1/2 * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 :=
by sorry

end triangle_proof_l418_41864


namespace x_greater_than_one_l418_41814

theorem x_greater_than_one (x : ℝ) (h : Real.log x > 0) : x > 1 := by
  sorry

end x_greater_than_one_l418_41814


namespace pi_irrational_less_than_neg_three_l418_41860

theorem pi_irrational_less_than_neg_three : 
  Irrational (-Real.pi) ∧ -Real.pi < -3 := by sorry

end pi_irrational_less_than_neg_three_l418_41860


namespace gcd_228_1995_l418_41885

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l418_41885


namespace village_panic_percentage_l418_41850

theorem village_panic_percentage (original : ℕ) (final : ℕ) : original = 7800 → final = 5265 → 
  (((original - original / 10) - final) / (original - original / 10) : ℚ) = 1/4 := by
  sorry

end village_panic_percentage_l418_41850


namespace largest_mersenne_prime_under_1000_l418_41849

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_mersenne_prime (m : ℕ) : Prop :=
  ∃ n : ℕ, is_prime n ∧ m = 2^n - 1 ∧ is_prime m

theorem largest_mersenne_prime_under_1000 :
  ∀ m : ℕ, is_mersenne_prime m → m < 1000 → m ≤ 127 :=
sorry

end largest_mersenne_prime_under_1000_l418_41849


namespace greatest_four_digit_number_with_conditions_l418_41857

/-- The greatest four-digit number that is two more than a multiple of 8 and four more than a multiple of 7 -/
def greatest_number : ℕ := 9990

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is two more than a multiple of 8 -/
def is_two_more_than_multiple_of_eight (n : ℕ) : Prop := ∃ k : ℕ, n = 8 * k + 2

/-- A number is four more than a multiple of 7 -/
def is_four_more_than_multiple_of_seven (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k + 4

theorem greatest_four_digit_number_with_conditions :
  is_four_digit greatest_number ∧
  is_two_more_than_multiple_of_eight greatest_number ∧
  is_four_more_than_multiple_of_seven greatest_number ∧
  ∀ n : ℕ, is_four_digit n →
    is_two_more_than_multiple_of_eight n →
    is_four_more_than_multiple_of_seven n →
    n ≤ greatest_number :=
by sorry

end greatest_four_digit_number_with_conditions_l418_41857


namespace two_non_congruent_triangles_l418_41879

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if two triangles are congruent -/
def is_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all triangles with perimeter 7 -/
def triangles_with_perimeter_7 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 7}

/-- The theorem to be proved -/
theorem two_non_congruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ triangles_with_perimeter_7 ∧
    t2 ∈ triangles_with_perimeter_7 ∧
    ¬ is_congruent t1 t2 ∧
    ∀ (t : IntTriangle),
      t ∈ triangles_with_perimeter_7 →
      is_congruent t t1 ∨ is_congruent t t2 :=
sorry

end two_non_congruent_triangles_l418_41879


namespace two_white_socks_cost_45_cents_l418_41884

/-- The cost of a single brown sock in cents -/
def brown_sock_cost : ℕ := 300 / 15

/-- The cost of two white socks in cents -/
def white_socks_cost : ℕ := brown_sock_cost + 25

theorem two_white_socks_cost_45_cents : white_socks_cost = 45 := by
  sorry

#eval white_socks_cost

end two_white_socks_cost_45_cents_l418_41884


namespace alligator_coins_l418_41825

def river_crossing (initial : ℚ) : ℚ := 
  ((((initial * 3 - 30) * 3 - 30) * 3 - 30) * 3 - 30)

theorem alligator_coins : 
  ∃ initial : ℚ, river_crossing initial = 10 ∧ initial = 1210 / 81 := by
sorry

end alligator_coins_l418_41825


namespace irreducible_fractions_divisibility_l418_41828

theorem irreducible_fractions_divisibility (a n : ℕ) (ha : a > 1) (hn : n > 1) :
  ∃ k : ℕ, Nat.totient (a^n - 1) = n * k := by
  sorry

end irreducible_fractions_divisibility_l418_41828


namespace sin_plus_cos_special_angle_l418_41896

/-- Given a point P(3, 4) on the terminal side of angle α, prove that sin α + cos α = 8/5 -/
theorem sin_plus_cos_special_angle (α : Real) :
  let P : Real × Real := (3, 4)
  (P.1 = 3 ∧ P.2 = 4) →  -- Point P has coordinates (3, 4)
  (P.1^2 + P.2^2 = 5^2) →  -- P is on the unit circle with radius 5
  (Real.sin α = P.2 / 5 ∧ Real.cos α = P.1 / 5) →  -- Definition of sin and cos for this point
  Real.sin α + Real.cos α = 8/5 := by
  sorry

end sin_plus_cos_special_angle_l418_41896


namespace no_solution_iff_k_eq_seven_l418_41873

theorem no_solution_iff_k_eq_seven :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 := by
  sorry

end no_solution_iff_k_eq_seven_l418_41873


namespace tan_period_l418_41898

/-- The period of y = tan(3x/4) is 4π/3 -/
theorem tan_period (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.tan (3 * x / 4)
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = 4 * Real.pi / 3 := by
  sorry

end tan_period_l418_41898


namespace twenty_percent_less_than_sixty_l418_41868

theorem twenty_percent_less_than_sixty (x : ℝ) : x + (1/3) * x = 48 → x = 36 := by
  sorry

end twenty_percent_less_than_sixty_l418_41868


namespace problem_solution_l418_41806

theorem problem_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 6 * x^3 + 12 * x * y = 2 * x^4 + 3 * x^3 * y) (h4 : y = x^2) : 
  x = (-1 + Real.sqrt 55) / 3 := by
sorry

end problem_solution_l418_41806


namespace absolute_value_problem_l418_41878

theorem absolute_value_problem (a b : ℝ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a + b| = 4) :
  ∃ (x : ℝ), |a - b| = x :=
sorry

end absolute_value_problem_l418_41878


namespace sufficient_not_necessary_condition_l418_41899

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > a → x > 2) ∧ (∃ x, x > 2 ∧ x ≤ a) ↔ a > 2 :=
sorry

end sufficient_not_necessary_condition_l418_41899


namespace equilateral_triangle_isosceles_points_l418_41815

/-- An equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : sorry

/-- A point in the 2D plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a triangle is isosceles -/
def is_isosceles (P Q R : Point) : Prop := sorry

/-- Predicate to check if a point is inside a triangle -/
def is_inside (P : Point) (triangle : EquilateralTriangle) : Prop := sorry

theorem equilateral_triangle_isosceles_points (ABC : EquilateralTriangle) :
  ∃ (points : Finset Point),
    points.card = 10 ∧
    ∀ P ∈ points,
      is_inside P ABC ∧
      is_isosceles P ABC.B ABC.C ∧
      is_isosceles P ABC.A ABC.B ∧
      is_isosceles P ABC.A ABC.C :=
sorry

end equilateral_triangle_isosceles_points_l418_41815


namespace integral_2x_minus_1_l418_41813

theorem integral_2x_minus_1 : ∫ x in (1:ℝ)..(2:ℝ), 2*x - 1 = 2 := by sorry

end integral_2x_minus_1_l418_41813


namespace polynomial_factorization_l418_41881

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 9) * (x^2 + 6*x + 1) := by
  sorry

end polynomial_factorization_l418_41881


namespace polyhedron_sum_l418_41829

/-- A convex polyhedron with triangular and pentagonal faces -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  T : ℕ  -- number of triangular faces
  P : ℕ  -- number of pentagonal faces
  euler : V - E + F = 2
  faces : F = 32
  face_types : F = T + P
  vertex_edges : 2 * E = V * (T + P)
  edge_count : 3 * T + 5 * P = 2 * E

/-- Theorem stating that P + T + V = 34 for the given convex polyhedron -/
theorem polyhedron_sum (poly : ConvexPolyhedron) : poly.P + poly.T + poly.V = 34 := by
  sorry


end polyhedron_sum_l418_41829


namespace A_is_largest_l418_41811

/-- The value of expression A -/
def A : ℚ := 3009 / 3008 + 3009 / 3010

/-- The value of expression B -/
def B : ℚ := 3011 / 3010 + 3011 / 3012

/-- The value of expression C -/
def C : ℚ := 3010 / 3009 + 3010 / 3011

/-- Theorem stating that A is the largest among A, B, and C -/
theorem A_is_largest : A > B ∧ A > C := by
  sorry

end A_is_largest_l418_41811


namespace distance_between_homes_l418_41894

/-- Proves the distance between Maxwell's and Brad's homes given their speeds and meeting point -/
theorem distance_between_homes
  (maxwell_speed : ℝ)
  (brad_speed : ℝ)
  (maxwell_distance : ℝ)
  (h1 : maxwell_speed = 2)
  (h2 : brad_speed = 3)
  (h3 : maxwell_distance = 26)
  (h4 : maxwell_distance / maxwell_speed = (total_distance - maxwell_distance) / brad_speed) :
  total_distance = 65 :=
by
  sorry

#check distance_between_homes

end distance_between_homes_l418_41894


namespace stratified_sampling_most_appropriate_school_staff_sampling_l418_41876

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents a population with subgroups -/
structure Population where
  total : ℕ
  subgroups : List ℕ
  h_sum : total = subgroups.sum

/-- Represents a sample -/
structure Sample where
  size : ℕ
  method : SamplingMethod

/-- Determines if a sample is representative of a population -/
def is_representative (pop : Population) (samp : Sample) : Prop :=
  samp.method = SamplingMethod.Stratified ∧ pop.subgroups.length > 1

/-- The main theorem stating that stratified sampling is most appropriate for a population with subgroups -/
theorem stratified_sampling_most_appropriate (pop : Population) (samp : Sample) 
    (h_subgroups : pop.subgroups.length > 1) : 
    is_representative pop samp ↔ samp.method = SamplingMethod.Stratified :=
  sorry

/-- The specific instance from the problem -/
def school_staff : Population :=
  { total := 160
  , subgroups := [120, 16, 24]
  , h_sum := by simp }

def staff_sample : Sample :=
  { size := 20
  , method := SamplingMethod.Stratified }

/-- The theorem applied to the specific instance -/
theorem school_staff_sampling : 
    is_representative school_staff staff_sample :=
  sorry

end stratified_sampling_most_appropriate_school_staff_sampling_l418_41876


namespace min_m_bound_l418_41812

theorem min_m_bound (a b : ℝ) (h1 : |a - b| ≤ 1) (h2 : |2 * a - 1| ≤ 1) :
  ∃ m : ℝ, (∀ a b : ℝ, |a - b| ≤ 1 → |2 * a - 1| ≤ 1 → |4 * a - 3 * b + 2| ≤ m) ∧
  (∀ m' : ℝ, (∀ a b : ℝ, |a - b| ≤ 1 → |2 * a - 1| ≤ 1 → |4 * a - 3 * b + 2| ≤ m') → m ≤ m') ∧
  m = 6 :=
sorry

end min_m_bound_l418_41812


namespace f_of_3_equals_5_l418_41847

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem f_of_3_equals_5 : f 3 = 5 := by
  sorry

end f_of_3_equals_5_l418_41847


namespace unique_corresponding_point_l418_41841

-- Define a square as a structure with a side length and a position
structure Square where
  sideLength : ℝ
  position : ℝ × ℝ

-- Define the problem setup
axiom larger_square : Square
axiom smaller_square : Square

-- The smaller square is entirely within the larger square
axiom smaller_inside_larger :
  smaller_square.position.1 ≥ larger_square.position.1 ∧
  smaller_square.position.1 + smaller_square.sideLength ≤ larger_square.position.1 + larger_square.sideLength ∧
  smaller_square.position.2 ≥ larger_square.position.2 ∧
  smaller_square.position.2 + smaller_square.sideLength ≤ larger_square.position.2 + larger_square.sideLength

-- The squares have the same area
axiom same_area : larger_square.sideLength^2 = smaller_square.sideLength^2

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- The theorem to be proved
theorem unique_corresponding_point :
  ∃! p : Point,
    (p.1 - larger_square.position.1) / larger_square.sideLength =
    (p.1 - smaller_square.position.1) / smaller_square.sideLength ∧
    (p.2 - larger_square.position.2) / larger_square.sideLength =
    (p.2 - smaller_square.position.2) / smaller_square.sideLength :=
  sorry

end unique_corresponding_point_l418_41841


namespace bisection_method_step_l418_41867

def f (x : ℝ) := x^5 + 8*x^3 - 1

theorem bisection_method_step (h1 : f 0 < 0) (h2 : f 0.5 > 0) :
  ∃ x₀ ∈ Set.Ioo 0 0.5, f x₀ = 0 ∧ 0.25 = (0 + 0.5) / 2 := by
  sorry

end bisection_method_step_l418_41867


namespace inequality_theorem_l418_41861

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) := by
  sorry

end inequality_theorem_l418_41861


namespace girl_squirrel_walnuts_l418_41817

theorem girl_squirrel_walnuts (initial : ℕ) (boy_adds : ℕ) (girl_eats : ℕ) (final : ℕ) :
  initial = 12 →
  boy_adds = 5 →
  girl_eats = 2 →
  final = 20 →
  ∃ girl_brings : ℕ, initial + boy_adds + girl_brings - girl_eats = final ∧ girl_brings = 5 :=
by sorry

end girl_squirrel_walnuts_l418_41817


namespace course_size_l418_41826

theorem course_size (a b c d : ℕ) (h1 : a + b + c + d = 800) 
  (h2 : a = 800 / 5) (h3 : b = 800 / 4) (h4 : c = 800 / 2) (h5 : d = 40) : 
  800 = 800 := by sorry

end course_size_l418_41826


namespace rooms_per_floor_l418_41816

theorem rooms_per_floor (total_earnings : ℕ) (hourly_rate : ℕ) (hours_per_room : ℕ) (num_floors : ℕ)
  (h1 : total_earnings = 3600)
  (h2 : hourly_rate = 15)
  (h3 : hours_per_room = 6)
  (h4 : num_floors = 4) :
  total_earnings / (hourly_rate * num_floors) = 10 := by
  sorry

#check rooms_per_floor

end rooms_per_floor_l418_41816


namespace apple_profit_percentage_l418_41891

/-- Calculates the total profit percentage for a shopkeeper selling apples -/
theorem apple_profit_percentage
  (total_apples : ℝ)
  (percent_sold_at_low_profit : ℝ)
  (percent_sold_at_high_profit : ℝ)
  (low_profit_rate : ℝ)
  (high_profit_rate : ℝ)
  (h1 : total_apples = 280)
  (h2 : percent_sold_at_low_profit = 0.4)
  (h3 : percent_sold_at_high_profit = 0.6)
  (h4 : low_profit_rate = 0.1)
  (h5 : high_profit_rate = 0.3)
  (h6 : percent_sold_at_low_profit + percent_sold_at_high_profit = 1) :
  let cost_price := 1
  let low_profit_quantity := percent_sold_at_low_profit * total_apples
  let high_profit_quantity := percent_sold_at_high_profit * total_apples
  let total_cost := total_apples * cost_price
  let low_profit_revenue := low_profit_quantity * cost_price * (1 + low_profit_rate)
  let high_profit_revenue := high_profit_quantity * cost_price * (1 + high_profit_rate)
  let total_revenue := low_profit_revenue + high_profit_revenue
  let total_profit := total_revenue - total_cost
  let profit_percentage := (total_profit / total_cost) * 100
  profit_percentage = 22 := by sorry

end apple_profit_percentage_l418_41891


namespace max_profit_is_180_l418_41835

/-- Represents a neighborhood with its characteristics --/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ
  pricePerBox : ℚ
  transportCost : ℚ

/-- Calculates the profit for a given neighborhood --/
def profit (n : Neighborhood) : ℚ :=
  n.homes * n.boxesPerHome * n.pricePerBox - n.transportCost

/-- Checks if the neighborhood is within the stock limit --/
def withinStockLimit (n : Neighborhood) (stockLimit : ℕ) : Prop :=
  n.homes * n.boxesPerHome ≤ stockLimit

/-- The main theorem stating the maximum profit --/
theorem max_profit_is_180 (stockLimit : ℕ) (A B C D : Neighborhood)
  (hStock : stockLimit = 50)
  (hA : A = { homes := 12, boxesPerHome := 3, pricePerBox := 3, transportCost := 10 })
  (hB : B = { homes := 8, boxesPerHome := 6, pricePerBox := 4, transportCost := 15 })
  (hC : C = { homes := 15, boxesPerHome := 2, pricePerBox := 5/2, transportCost := 5 })
  (hD : D = { homes := 5, boxesPerHome := 8, pricePerBox := 5, transportCost := 20 })
  (hAStock : withinStockLimit A stockLimit)
  (hBStock : withinStockLimit B stockLimit)
  (hCStock : withinStockLimit C stockLimit)
  (hDStock : withinStockLimit D stockLimit) :
  (max (profit A) (max (profit B) (max (profit C) (profit D)))) = 180 :=
sorry

end max_profit_is_180_l418_41835


namespace pages_per_sheet_calculation_l418_41886

/-- The number of stories John writes per week -/
def stories_per_week : ℕ := 3

/-- The number of pages in each story -/
def pages_per_story : ℕ := 50

/-- The number of weeks John writes -/
def weeks : ℕ := 12

/-- The number of reams of paper John uses over 12 weeks -/
def reams_used : ℕ := 3

/-- The number of sheets in each ream of paper -/
def sheets_per_ream : ℕ := 500

/-- Calculate the number of pages each sheet of paper can hold -/
def pages_per_sheet : ℕ := 1

theorem pages_per_sheet_calculation :
  pages_per_sheet = 1 :=
by sorry

end pages_per_sheet_calculation_l418_41886


namespace neglart_hands_count_l418_41893

/-- Represents a race on planet Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of toes on each hand for a given race -/
def toes_per_hand (race : Race) : ℕ :=
  match race with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of hands for Hoopits -/
def hoopit_hands : ℕ := 4

/-- Number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating the number of hands each Neglart has -/
theorem neglart_hands_count :
  ∃ (neglart_hands : ℕ),
    neglart_hands * neglart_students * toes_per_hand Race.Neglart +
    hoopit_hands * hoopit_students * toes_per_hand Race.Hoopit = total_toes ∧
    neglart_hands = 5 := by
  sorry

end neglart_hands_count_l418_41893


namespace no_integer_square_root_l418_41871

theorem no_integer_square_root : ¬ ∃ (y : ℤ) (b : ℤ), y^4 + 8*y^3 + 18*y^2 + 10*y + 41 = b^2 := by
  sorry

end no_integer_square_root_l418_41871


namespace range_of_a_l418_41880

-- Define the sets A and B
def A : Set ℝ := Set.Ioo 1 4
def B (a : ℝ) : Set ℝ := Set.Ioo (2 * a) (a + 1)

-- State the theorem
theorem range_of_a (a : ℝ) (h : a < 1) :
  B a ⊆ A → 1/2 ≤ a ∧ a < 1 := by
sorry

end range_of_a_l418_41880


namespace expand_expression_l418_41892

theorem expand_expression (y z : ℝ) : 
  -2 * (5 * y^3 - 3 * y^2 * z + 4 * y * z^2 - z^3) = 
  -10 * y^3 + 6 * y^2 * z - 8 * y * z^2 + 2 * z^3 := by
sorry

end expand_expression_l418_41892


namespace limeade_calories_l418_41801

-- Define the components of limeade
def lime_juice_weight : ℝ := 150
def sugar_weight : ℝ := 200
def water_weight : ℝ := 450

-- Define calorie content per 100g
def lime_juice_calories_per_100g : ℝ := 20
def sugar_calories_per_100g : ℝ := 396
def water_calories_per_100g : ℝ := 0

-- Define the weight of limeade we want to calculate calories for
def limeade_sample_weight : ℝ := 300

-- Theorem statement
theorem limeade_calories : 
  let total_weight := lime_juice_weight + sugar_weight + water_weight
  let total_calories := (lime_juice_calories_per_100g * lime_juice_weight / 100) + 
                        (sugar_calories_per_100g * sugar_weight / 100) + 
                        (water_calories_per_100g * water_weight / 100)
  (total_calories * limeade_sample_weight / total_weight) = 308.25 := by
  sorry

end limeade_calories_l418_41801


namespace floor_sum_example_l418_41877

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l418_41877


namespace area_between_curves_l418_41805

-- Define the two functions
def f (y : ℝ) : ℝ := 4 - (y - 1)^2
def g (y : ℝ) : ℝ := y^2 - 4*y + 3

-- Define the bounds of integration
def lower_bound : ℝ := 0
def upper_bound : ℝ := 3

-- State the theorem
theorem area_between_curves : 
  (∫ y in lower_bound..upper_bound, f y - g y) = 9 := by sorry

end area_between_curves_l418_41805


namespace alcohol_mixture_proof_l418_41837

/-- Proves that mixing 200 mL of 10% alcohol solution with 600 mL of 30% alcohol solution results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 200
  let x_concentration : ℝ := 0.10
  let y_volume : ℝ := 600
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  (total_alcohol / total_volume) = target_concentration :=
by sorry

end alcohol_mixture_proof_l418_41837


namespace rectangle_area_l418_41869

theorem rectangle_area (perimeter : ℝ) (h1 : perimeter = 160) : ∃ (length width : ℝ),
  length = 4 * width ∧
  2 * (length + width) = perimeter ∧
  length * width = 1024 := by
  sorry

end rectangle_area_l418_41869


namespace dan_onions_l418_41862

/-- The number of onions grown by Nancy, Dan, and Mike -/
structure OnionGrowth where
  nancy : ℕ
  dan : ℕ
  mike : ℕ

/-- The total number of onions grown -/
def total_onions (g : OnionGrowth) : ℕ :=
  g.nancy + g.dan + g.mike

/-- Theorem: Dan grew 9 onions -/
theorem dan_onions :
  ∀ g : OnionGrowth,
    g.nancy = 2 →
    g.mike = 4 →
    total_onions g = 15 →
    g.dan = 9 := by
  sorry

end dan_onions_l418_41862


namespace perimeter_is_24_l418_41840

/-- A right triangle ABC with specific properties -/
structure RightTriangleABC where
  -- Points A, B, and C
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- ABC is a right triangle with right angle at B
  is_right_triangle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  -- Angle BAC equals angle BCA
  angle_equality : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 
                   (C.1 - B.1) * (C.1 - A.1) + (C.2 - B.2) * (C.2 - A.2)
  -- Length of AB is 9
  AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 9
  -- Length of BC is 6
  BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 6

/-- The perimeter of the right triangle ABC is 24 -/
theorem perimeter_is_24 (t : RightTriangleABC) : 
  Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) +
  Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) +
  Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) = 24 := by
  sorry


end perimeter_is_24_l418_41840


namespace sum_of_relatively_prime_integers_l418_41872

theorem sum_of_relatively_prime_integers (n : ℤ) (h : n ≥ 7) :
  ∃ a b : ℤ, n = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 := by
  sorry

end sum_of_relatively_prime_integers_l418_41872


namespace chris_money_before_birthday_l418_41803

/-- Represents the amount of money Chris had before his birthday. -/
def money_before_birthday : ℕ := sorry

/-- The amount Chris received from his grandmother. -/
def grandmother_gift : ℕ := 25

/-- The amount Chris received from his aunt and uncle. -/
def aunt_uncle_gift : ℕ := 20

/-- The amount Chris received from his parents. -/
def parents_gift : ℕ := 75

/-- The total amount Chris has after receiving all gifts. -/
def total_after_gifts : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday. -/
theorem chris_money_before_birthday :
  money_before_birthday = total_after_gifts - (grandmother_gift + aunt_uncle_gift + parents_gift) :=
by sorry

end chris_money_before_birthday_l418_41803


namespace ball_probabilities_l418_41827

/-- Represents the color of a ball -/
inductive BallColor
| Black
| White

/-- Represents the bag of balls -/
structure Bag :=
  (black : ℕ)
  (white : ℕ)

/-- The probability of drawing a white ball on the third draw with replacement -/
def prob_white_third_with_replacement (bag : Bag) : ℚ :=
  bag.white / (bag.black + bag.white)

/-- The probability of drawing a white ball only on the third draw with replacement -/
def prob_white_only_third_with_replacement (bag : Bag) : ℚ :=
  (bag.black / (bag.black + bag.white))^2 * (bag.white / (bag.black + bag.white))

/-- The probability of drawing a white ball on the third draw without replacement -/
def prob_white_third_without_replacement (bag : Bag) : ℚ :=
  (bag.white * (bag.black * (bag.black - 1) + 2 * bag.black * bag.white + bag.white * (bag.white - 1))) /
  ((bag.black + bag.white) * (bag.black + bag.white - 1) * (bag.black + bag.white - 2))

/-- The probability of drawing a white ball only on the third draw without replacement -/
def prob_white_only_third_without_replacement (bag : Bag) : ℚ :=
  (bag.black * (bag.black - 1) * bag.white) /
  ((bag.black + bag.white) * (bag.black + bag.white - 1) * (bag.black + bag.white - 2))

theorem ball_probabilities (bag : Bag) (h : bag.black = 3 ∧ bag.white = 2) :
  prob_white_third_with_replacement bag = 2/5 ∧
  prob_white_only_third_with_replacement bag = 18/125 ∧
  prob_white_third_without_replacement bag = 2/5 ∧
  prob_white_only_third_without_replacement bag = 1/5 := by
  sorry

end ball_probabilities_l418_41827


namespace monochromatic_isosceles_independent_of_coloring_l418_41831

/-- Represents a regular polygon with 6n+1 sides and a coloring of its vertices -/
structure ColoredPolygon (n : ℕ) where
  k : ℕ
  h : ℕ := 6 * n + 1 - k
  k_valid : k ≤ 6 * n + 1

/-- Counts the number of monochromatic isosceles triangles in a colored polygon -/
def monochromaticIsoscelesCount (p : ColoredPolygon n) : ℚ :=
  (1 / 2) * (p.h * (p.h - 1) + p.k * (p.k - 1) - p.k * p.h)

/-- Theorem stating that the number of monochromatic isosceles triangles is independent of coloring -/
theorem monochromatic_isosceles_independent_of_coloring (n : ℕ) :
  ∀ p q : ColoredPolygon n, monochromaticIsoscelesCount p = monochromaticIsoscelesCount q :=
sorry

end monochromatic_isosceles_independent_of_coloring_l418_41831


namespace one_female_selection_l418_41856

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of male students in group A -/
def maleA : ℕ := 5

/-- The number of female students in group A -/
def femaleA : ℕ := 3

/-- The number of male students in group B -/
def maleB : ℕ := 6

/-- The number of female students in group B -/
def femaleB : ℕ := 2

/-- The number of students to be selected from each group -/
def selectPerGroup : ℕ := 2

/-- The total number of ways to select exactly one female student among 4 chosen students -/
theorem one_female_selection : 
  (choose femaleA 1 * choose maleA 1 * choose maleB selectPerGroup) + 
  (choose femaleB 1 * choose maleB 1 * choose maleA selectPerGroup) = 345 := by
  sorry

end one_female_selection_l418_41856


namespace multiple_of_960_l418_41845

theorem multiple_of_960 (a : ℤ) 
  (h1 : ∃ k : ℤ, a = 10 * k + 4) 
  (h2 : ¬ (∃ m : ℤ, a = 4 * m)) : 
  ∃ n : ℤ, a * (a^2 - 1) * (a^2 - 4) = 960 * n :=
sorry

end multiple_of_960_l418_41845


namespace y_value_l418_41895

theorem y_value : ∃ y : ℝ, y ≠ 0 ∧ y = 2 * (1 / y) * (-y) - 4 → y = -6 := by
  sorry

end y_value_l418_41895


namespace reciprocal_sum_of_roots_l418_41842

theorem reciprocal_sum_of_roots (γ δ : ℝ) : 
  (∃ r s : ℝ, 7 * r^2 + 4 * r + 9 = 0 ∧ 
              7 * s^2 + 4 * s + 9 = 0 ∧ 
              γ = 1 / r ∧ 
              δ = 1 / s) → 
  γ + δ = -4/9 := by
sorry

end reciprocal_sum_of_roots_l418_41842


namespace bird_families_flew_away_l418_41875

/-- The number of bird families that flew away is equal to the difference between
    the total number of bird families and the number of bird families left. -/
theorem bird_families_flew_away (total : ℕ) (left : ℕ) (flew_away : ℕ) 
    (h1 : total = 67) (h2 : left = 35) (h3 : flew_away = total - left) : 
    flew_away = 32 := by
  sorry

end bird_families_flew_away_l418_41875


namespace symmetric_line_y_axis_neg_2x_minus_3_l418_41808

/-- Given a line with equation y = mx + b, this function returns the equation
    of the line symmetric to it with respect to the y-axis -/
def symmetricLineYAxis (m : ℝ) (b : ℝ) : ℝ → ℝ := fun x ↦ -m * x + b

theorem symmetric_line_y_axis_neg_2x_minus_3 :
  symmetricLineYAxis (-2) (-3) = fun x ↦ 2 * x - 3 := by sorry

end symmetric_line_y_axis_neg_2x_minus_3_l418_41808


namespace square_of_square_plus_eight_l418_41819

theorem square_of_square_plus_eight : (4^2 + 8)^2 = 576 := by
  sorry

end square_of_square_plus_eight_l418_41819


namespace chocolate_division_l418_41834

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_for_shaina : ℕ) : 
  total_chocolate = 60 / 7 →
  num_piles = 5 →
  piles_for_shaina = 2 →
  piles_for_shaina * (total_chocolate / num_piles) = 24 / 7 := by
sorry

end chocolate_division_l418_41834


namespace min_value_problem_l418_41852

/-- Given positive real numbers a, b, c, and a function f with minimum value 4, 
    prove the sum of a, b, c is 4 and find the minimum value of a quadratic expression. -/
theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 4) : 
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) :=
by sorry

end min_value_problem_l418_41852


namespace base_conversion_2014_to_base_9_l418_41844

theorem base_conversion_2014_to_base_9 :
  2014 = 2 * (9^3) + 6 * (9^2) + 7 * (9^1) + 7 * (9^0) :=
by sorry

end base_conversion_2014_to_base_9_l418_41844


namespace difference_of_squares_special_case_l418_41889

theorem difference_of_squares_special_case : (733 : ℤ) * 733 - 732 * 734 = 1 := by
  sorry

end difference_of_squares_special_case_l418_41889


namespace line_tangent_to_circle_l418_41890

/-- The line x = my + 2 is tangent to the circle x^2 + 2x + y^2 + 2y = 0 if and only if m = 1 or m = -7 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x = m * y + 2 → x^2 + 2*x + y^2 + 2*y ≠ 0) ∨
  (∃! x y : ℝ, x = m * y + 2 ∧ x^2 + 2*x + y^2 + 2*y = 0) ↔ 
  m = 1 ∨ m = -7 := by
sorry

end line_tangent_to_circle_l418_41890


namespace coprime_set_properties_l418_41823

-- Define the set M
def M (a b : ℕ) : Set ℤ :=
  {z : ℤ | ∃ (x y : ℕ), z = a * x + b * y}

-- State the theorem
theorem coprime_set_properties (a b : ℕ) (h : Nat.Coprime a b) :
  -- Part 1: The largest integer not in M is ab - a - b
  (∀ z : ℤ, z ∉ M a b → z ≤ a * b - a - b) ∧
  (a * b - a - b : ℤ) ∉ M a b ∧
  -- Part 2: For any integer n, exactly one of n and (ab - a - b - n) is in M
  (∀ n : ℤ, (n ∈ M a b ↔ (a * b - a - b - n) ∉ M a b)) := by
  sorry

end coprime_set_properties_l418_41823


namespace coefficient_x_cubed_is_22_l418_41853

/-- The first polynomial -/
def p1 (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

/-- The second polynomial -/
def p2 (x : ℝ) : ℝ := 3*x^3 - 4*x^2 + x + 6

/-- The product of the two polynomials -/
def product (x : ℝ) : ℝ := p1 x * p2 x

/-- Theorem stating that the coefficient of x^3 in the product is 22 -/
theorem coefficient_x_cubed_is_22 : 
  ∃ (a b c d e : ℝ), product = fun x ↦ 22 * x^3 + a * x^4 + b * x^2 + c * x + d * x^5 + e :=
sorry

end coefficient_x_cubed_is_22_l418_41853


namespace gas_price_increase_l418_41800

theorem gas_price_increase (P : ℝ) (h : P > 0) : 
  let first_increase := 0.15
  let consumption_reduction := 0.20948616600790515
  let second_increase := 0.1
  let final_price := P * (1 + first_increase) * (1 + second_increase)
  let reduced_consumption := 1 - consumption_reduction
  reduced_consumption * final_price = P :=
by sorry

end gas_price_increase_l418_41800


namespace steves_final_height_l418_41804

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Calculates the final height in inches after growth -/
def final_height (initial_feet : ℕ) (initial_inches : ℕ) (growth : ℕ) : ℕ :=
  feet_inches_to_inches initial_feet initial_inches + growth

/-- Theorem: Steve's final height is 72 inches -/
theorem steves_final_height :
  final_height 5 6 6 = 72 := by
  sorry

end steves_final_height_l418_41804


namespace short_trees_after_planting_park_short_trees_l418_41832

/-- The number of short trees in a park after planting new trees -/
def total_short_trees (initial_short_trees new_short_trees : ℕ) : ℕ :=
  initial_short_trees + new_short_trees

/-- Theorem: The total number of short trees after planting is the sum of initial and new short trees -/
theorem short_trees_after_planting 
  (initial_short_trees : ℕ) 
  (new_short_trees : ℕ) : 
  total_short_trees initial_short_trees new_short_trees = initial_short_trees + new_short_trees := by
  sorry

/-- Application to the specific problem -/
theorem park_short_trees : total_short_trees 3 9 = 12 := by
  sorry

end short_trees_after_planting_park_short_trees_l418_41832


namespace greatest_x_given_lcm_l418_41820

theorem greatest_x_given_lcm (x : ℕ) : 
  Nat.lcm x (Nat.lcm 15 21) = 105 → x ≤ 105 :=
by sorry

end greatest_x_given_lcm_l418_41820


namespace max_profit_at_800_l418_41836

/-- Price function for desk orders -/
def P (x : ℕ) : ℚ :=
  if x ≤ 100 then 80
  else 82 - 0.02 * x

/-- Profit function for desk orders -/
def f (x : ℕ) : ℚ :=
  if x ≤ 100 then 30 * x
  else (32 * x - 0.02 * x^2)

/-- Theorem stating the maximum profit and corresponding order quantity -/
theorem max_profit_at_800 :
  (∀ x : ℕ, 0 < x ∧ x ≤ 1000 → f x ≤ f 800) ∧
  f 800 = 12800 :=
sorry

end max_profit_at_800_l418_41836


namespace time_reduction_percentage_l418_41822

/-- Calculates the time reduction percentage when increasing speed from 60 km/h to 86 km/h for a journey that initially takes 30 minutes. -/
theorem time_reduction_percentage 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (new_speed : ℝ) 
  (h1 : initial_speed = 60) 
  (h2 : initial_time = 30) 
  (h3 : new_speed = 86) : 
  ∃ (reduction_percentage : ℝ), 
    (abs (reduction_percentage - 30.23) < 0.01) ∧ 
    (reduction_percentage = (1 - (initial_speed * initial_time) / (new_speed * initial_time)) * 100) :=
by sorry

end time_reduction_percentage_l418_41822


namespace dans_earnings_difference_l418_41887

/-- Calculates the difference in earnings between two sets of tasks -/
def earningsDifference (numTasks1 : ℕ) (rate1 : ℚ) (numTasks2 : ℕ) (rate2 : ℚ) : ℚ :=
  numTasks1 * rate1 - numTasks2 * rate2

/-- Proves that the difference in earnings between 400 tasks at $0.25 each and 5 tasks at $2.00 each is $90 -/
theorem dans_earnings_difference :
  earningsDifference 400 (25 / 100) 5 2 = 90 := by
  sorry

end dans_earnings_difference_l418_41887


namespace sqrt_x_minus_one_real_l418_41809

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by
  sorry

end sqrt_x_minus_one_real_l418_41809


namespace percentage_of_day_l418_41824

theorem percentage_of_day (hours_in_day : ℝ) (percentage : ℝ) (result : ℝ) : 
  hours_in_day = 24 →
  percentage = 29.166666666666668 →
  result = 7 →
  (percentage / 100) * hours_in_day = result :=
by sorry

end percentage_of_day_l418_41824


namespace t_shape_perimeter_l418_41882

/-- A T-shaped figure composed of squares -/
structure TShape where
  side_length : ℝ
  is_t_shaped : Bool
  horizontal_squares : ℕ
  vertical_squares : ℕ

/-- Calculate the perimeter of a T-shaped figure -/
def perimeter (t : TShape) : ℝ :=
  sorry

/-- Theorem: The perimeter of the specific T-shaped figure is 18 -/
theorem t_shape_perimeter :
  ∃ (t : TShape),
    t.side_length = 2 ∧
    t.is_t_shaped = true ∧
    t.horizontal_squares = 3 ∧
    t.vertical_squares = 1 ∧
    perimeter t = 18 :=
  sorry

end t_shape_perimeter_l418_41882


namespace transform_F_coordinates_l418_41839

/-- Reflection over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflection over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflection over the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- The initial coordinates of point F -/
def F : ℝ × ℝ := (1, 0)

theorem transform_F_coordinates :
  (reflect_y_eq_x ∘ reflect_y ∘ reflect_x) F = (0, -1) := by
  sorry

end transform_F_coordinates_l418_41839


namespace kate_red_bouncy_balls_l418_41843

/-- The number of packs of yellow bouncy balls Kate bought -/
def yellow_packs : ℕ := 6

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 18

/-- The difference in the number of red and yellow bouncy balls -/
def red_yellow_diff : ℕ := 18

/-- The number of packs of red bouncy balls Kate bought -/
def red_packs : ℕ := 7

theorem kate_red_bouncy_balls :
  red_packs * balls_per_pack = yellow_packs * balls_per_pack + red_yellow_diff :=
by sorry

end kate_red_bouncy_balls_l418_41843


namespace certain_number_proof_l418_41851

theorem certain_number_proof (x : ℝ) : x / 14.5 = 179 → x = 2595.5 := by
  sorry

end certain_number_proof_l418_41851
