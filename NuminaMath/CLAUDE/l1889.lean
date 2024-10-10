import Mathlib

namespace paths_equal_combinations_correct_number_of_paths_l1889_188903

/-- The number of paths from (0,0) to (8,8) on an 8x8 grid -/
def number_of_paths : ℕ := 12870

/-- The size of the grid -/
def grid_size : ℕ := 8

/-- The total number of steps required to reach from (0,0) to (8,8) -/
def total_steps : ℕ := 16

/-- The number of right steps required -/
def right_steps : ℕ := 8

/-- The number of up steps required -/
def up_steps : ℕ := 8

/-- Theorem stating that the number of paths from (0,0) to (8,8) on an 8x8 grid
    is equal to the number of ways to choose 8 up steps out of 16 total steps -/
theorem paths_equal_combinations :
  number_of_paths = Nat.choose total_steps up_steps :=
sorry

/-- Theorem stating that the number of paths is correct -/
theorem correct_number_of_paths :
  number_of_paths = 12870 :=
sorry

end paths_equal_combinations_correct_number_of_paths_l1889_188903


namespace log_product_equality_l1889_188932

theorem log_product_equality : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 2 := by
  sorry

end log_product_equality_l1889_188932


namespace hyperbola_t_range_l1889_188933

-- Define the curve C
def curve_C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - t) + y^2 / (t - 1) = 1}

-- Define what it means for a curve to be a hyperbola
def is_hyperbola (C : Set (ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem hyperbola_t_range (t : ℝ) :
  is_hyperbola (curve_C t) → t < 1 ∨ t > 4 := by
  sorry

end hyperbola_t_range_l1889_188933


namespace max_identical_end_digits_of_square_l1889_188913

theorem max_identical_end_digits_of_square (n : ℕ) (h : n % 10 ≠ 0) :
  ∀ k : ℕ, k > 4 → ∃ d : ℕ, d < 10 ∧ (n^2) % (10^k) ≠ d * ((10^k - 1) / 9) :=
sorry

end max_identical_end_digits_of_square_l1889_188913


namespace root_sum_equals_three_l1889_188914

noncomputable section

-- Define the logarithm base 10 function
def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the equations for x₁ and x₂
def equation1 (x : ℝ) : Prop := x + log10 x = 3
def equation2 (x : ℝ) : Prop := x + 10^x = 3

-- State the theorem
theorem root_sum_equals_three 
  (x₁ x₂ : ℝ) 
  (h1 : equation1 x₁) 
  (h2 : equation2 x₂) : 
  x₁ + x₂ = 3 := by sorry

end

end root_sum_equals_three_l1889_188914


namespace sum_of_reciprocals_equals_one_l1889_188959

theorem sum_of_reciprocals_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1/x + 1/y = 1 := by
sorry

end sum_of_reciprocals_equals_one_l1889_188959


namespace z_in_fourth_quadrant_l1889_188995

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 1

-- Define what it means for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  Complex.re z > 0 ∧ Complex.im z < 0

-- State the theorem
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, z_condition z ∧ in_fourth_quadrant z := by sorry

end z_in_fourth_quadrant_l1889_188995


namespace solution_set_f_geq_1_max_value_g_l1889_188917

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

-- Define the function g
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem for the maximum value of g(x)
theorem max_value_g :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

end solution_set_f_geq_1_max_value_g_l1889_188917


namespace power_equation_solution_l1889_188927

theorem power_equation_solution (m : ℝ) : 2^m = (64 : ℝ)^(1/3) → m = 2 := by
  sorry

end power_equation_solution_l1889_188927


namespace target_hitting_probability_l1889_188949

theorem target_hitting_probability (miss_prob : ℝ) (hit_prob : ℝ) :
  miss_prob = 0.20 →
  hit_prob = 1 - miss_prob →
  hit_prob = 0.80 :=
by
  sorry

end target_hitting_probability_l1889_188949


namespace sum_of_roots_equals_negative_two_ninths_l1889_188941

-- Define the function f
def f (x : ℝ) : ℝ := (3*x)^2 + 2*(3*x) + 2

-- State the theorem
theorem sum_of_roots_equals_negative_two_ninths :
  ∃ (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ f z₁ = 10 ∧ f z₂ = 10 ∧ z₁ + z₂ = -2/9 :=
sorry

end sum_of_roots_equals_negative_two_ninths_l1889_188941


namespace divisibility_property_l1889_188992

theorem divisibility_property (y : ℕ) (hy : y ≠ 0) :
  (y - 1) ∣ (y^(y^2) - 2*y^(y+1) + 1) := by
  sorry

end divisibility_property_l1889_188992


namespace pics_per_album_l1889_188937

-- Define the given conditions
def phone_pics : ℕ := 35
def camera_pics : ℕ := 5
def num_albums : ℕ := 5

-- Define the total number of pictures
def total_pics : ℕ := phone_pics + camera_pics

-- Theorem to prove
theorem pics_per_album : total_pics / num_albums = 8 := by
  sorry

end pics_per_album_l1889_188937


namespace triangle_cut_range_l1889_188931

/-- Given a triangle with side lengths 4, 5, and 6,
    if x is cut off from all sides resulting in an obtuse triangle,
    then 1 < x < 3 -/
theorem triangle_cut_range (x : ℝ) : 
  let a := 4 - x
  let b := 5 - x
  let c := 6 - x
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (a^2 + b^2 - c^2) / (2 * a * b) < 0 →
  1 < x ∧ x < 3 :=
by sorry


end triangle_cut_range_l1889_188931


namespace horner_method_evaluation_l1889_188955

/-- Horner's Method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => a + x * acc) 0

/-- The polynomial function -/
def f (x : ℝ) : ℝ :=
  1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem horner_method_evaluation :
  let coeffs := [0.00833, 0.04167, 0.16667, 0.5, 1, 1]
  abs (horner_eval coeffs (-0.2) - f (-0.2)) < 1e-5 ∧
  abs (horner_eval coeffs (-0.2) - 0.00427) < 1e-5 := by
  sorry

end horner_method_evaluation_l1889_188955


namespace quadratic_inequality_solution_l1889_188912

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx - 2 > 0 ↔ -4 < x ∧ x < 1) → a + b = 2 := by
  sorry

end quadratic_inequality_solution_l1889_188912


namespace gotham_street_homes_l1889_188981

theorem gotham_street_homes (total_homes : ℚ) : 
  let termite_ridden := (1 / 3 : ℚ) * total_homes
  let collapsing := (1 / 4 : ℚ) * termite_ridden
  termite_ridden - collapsing = (1 / 4 : ℚ) * total_homes :=
by sorry

end gotham_street_homes_l1889_188981


namespace complex_equation_solution_l1889_188971

theorem complex_equation_solution (z : ℂ) : 
  (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I :=
by sorry

end complex_equation_solution_l1889_188971


namespace num_correct_statements_is_zero_l1889_188919

/-- Definition of a frustum -/
structure Frustum where
  has_parallel_bases : Bool
  lateral_edges_converge : Bool

/-- The three statements about frustums -/
def statement1 (f : Frustum) : Prop :=
  true -- We don't need to define this precisely as it's always false

def statement2 (f : Frustum) : Prop :=
  f.has_parallel_bases

def statement3 (f : Frustum) : Prop :=
  f.has_parallel_bases

/-- Theorem: The number of correct statements is 0 -/
theorem num_correct_statements_is_zero : 
  (∀ f : Frustum, ¬statement1 f) ∧ 
  (∀ f : Frustum, f.has_parallel_bases ∧ f.lateral_edges_converge → statement2 f) ∧
  (∀ f : Frustum, f.has_parallel_bases ∧ f.lateral_edges_converge → statement3 f) →
  (¬∃ f : Frustum, statement1 f) ∧ 
  (¬∃ f : Frustum, statement2 f) ∧ 
  (¬∃ f : Frustum, statement3 f) :=
by
  sorry

#check num_correct_statements_is_zero

end num_correct_statements_is_zero_l1889_188919


namespace smallest_cube_divisible_by_primes_l1889_188998

theorem smallest_cube_divisible_by_primes (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r → p ≠ 1 → q ≠ 1 → r ≠ 1 →
  (pqr2_cube : ℕ) → pqr2_cube = (p * q * r^2)^3 →
  (∀ m : ℕ, m^3 ∣ p^2 * q^3 * r^5 → m^3 ≥ pqr2_cube) :=
by sorry

end smallest_cube_divisible_by_primes_l1889_188998


namespace banana_distribution_l1889_188939

theorem banana_distribution (total_bananas : ℕ) : 
  (∀ (children : ℕ), 
    (children * 2 = total_bananas) →
    ((children - 160) * 4 = total_bananas)) →
  ∃ (actual_children : ℕ), actual_children = 320 := by
  sorry

end banana_distribution_l1889_188939


namespace circle_circumference_with_inscribed_rectangle_l1889_188979

/-- Given a circle with an inscribed rectangle of dimensions 9 cm by 12 cm,
    the circumference of the circle is 15π cm. -/
theorem circle_circumference_with_inscribed_rectangle :
  ∀ (C : ℝ → ℝ → Prop) (r : ℝ),
    (∃ (x y : ℝ), C x y ∧ x^2 + y^2 = r^2 ∧ x = 9 ∧ y = 12) →
    2 * π * r = 15 * π :=
by sorry

end circle_circumference_with_inscribed_rectangle_l1889_188979


namespace mary_found_47_shells_l1889_188997

/-- The number of seashells Sam found -/
def sam_shells : ℕ := 18

/-- The total number of seashells Sam and Mary found together -/
def total_shells : ℕ := 65

/-- The number of seashells Mary found -/
def mary_shells : ℕ := total_shells - sam_shells

theorem mary_found_47_shells : mary_shells = 47 := by
  sorry

end mary_found_47_shells_l1889_188997


namespace buddy_fraction_l1889_188969

theorem buddy_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  (n : ℚ) / 3 = (2 : ℚ) * s / 5 → 
  ((n : ℚ) / 3 + (2 : ℚ) * s / 5) / (n + s : ℚ) = 4 / 11 := by
  sorry

end buddy_fraction_l1889_188969


namespace discount_savings_l1889_188996

/-- Given a purchase with a 10% discount, calculate the amount saved -/
theorem discount_savings (purchase_price : ℝ) (discount_rate : ℝ) (savings : ℝ) : 
  purchase_price = 100 →
  discount_rate = 0.1 →
  savings = purchase_price * discount_rate →
  savings = 10 := by
  sorry

end discount_savings_l1889_188996


namespace omega_range_l1889_188962

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃ a b : ℝ, π ≤ a ∧ a < b ∧ b ≤ 2*π ∧ Real.sin (ω * a) + Real.sin (ω * b) = 2) →
  (ω ∈ Set.Icc (9/4) (5/2) ∪ Set.Ici (13/4)) :=
by sorry

end omega_range_l1889_188962


namespace power_function_odd_l1889_188970

def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ n : ℤ, ∀ x : ℝ, f x = x ^ n

def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem power_function_odd (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 1 = 3) :
  isOddFunction f := by
  sorry

end power_function_odd_l1889_188970


namespace tank_capacities_l1889_188988

theorem tank_capacities (x y z : ℚ) : 
  x + y + z = 1620 →
  z = x + (1/5) * y →
  z = y + (1/3) * x →
  x = 540 ∧ y = 450 ∧ z = 630 := by
sorry

end tank_capacities_l1889_188988


namespace democrat_ratio_l1889_188918

theorem democrat_ratio (total_participants : ℕ) 
  (female_democrats : ℕ) (total_democrats : ℕ) :
  total_participants = 750 →
  female_democrats = 125 →
  total_democrats = total_participants / 3 →
  female_democrats * 2 ≤ total_participants →
  (total_democrats - female_democrats) * 4 = 
    total_participants - female_democrats * 2 := by
  sorry

end democrat_ratio_l1889_188918


namespace last_guard_hours_l1889_188999

/-- Represents the number of hours in a night shift -/
def total_hours : ℕ := 9

/-- Represents the number of guards -/
def num_guards : ℕ := 4

/-- Represents the hours taken by the first guard -/
def first_guard_hours : ℕ := 3

/-- Represents the hours taken by each middle guard -/
def middle_guard_hours : ℕ := 2

/-- Represents the number of middle guards -/
def num_middle_guards : ℕ := 2

theorem last_guard_hours :
  total_hours - (first_guard_hours + num_middle_guards * middle_guard_hours) = 2 := by
  sorry

end last_guard_hours_l1889_188999


namespace circle_op_equation_solution_l1889_188924

-- Define the € operation
def circle_op (x y : ℝ) : ℝ := 3 * x * y

-- State the theorem
theorem circle_op_equation_solution :
  ∀ z : ℝ, circle_op (circle_op 4 5) z = 540 → z = 3 := by
  sorry

end circle_op_equation_solution_l1889_188924


namespace smallest_AC_l1889_188989

/-- Triangle ABC with point D on AC --/
structure Triangle :=
  (AC : ℕ)
  (CD : ℕ)
  (BD : ℝ)

/-- Conditions for the triangle --/
def ValidTriangle (t : Triangle) : Prop :=
  t.AC = t.AC  -- AB = AC (isosceles)
  ∧ t.CD ≤ t.AC  -- D is on AC
  ∧ t.BD ^ 2 = 85  -- BD² = 85
  ∧ t.AC ^ 2 = (t.AC - t.CD) ^ 2 + 85  -- Pythagorean theorem

/-- The smallest possible AC value is 11 --/
theorem smallest_AC : 
  ∀ t : Triangle, ValidTriangle t → t.AC ≥ 11 ∧ ∃ t' : Triangle, ValidTriangle t' ∧ t'.AC = 11 :=
sorry

end smallest_AC_l1889_188989


namespace valid_scheduling_orders_l1889_188908

def number_of_lecturers : ℕ := 7

def number_of_dependencies : ℕ := 2

theorem valid_scheduling_orders :
  (number_of_lecturers.factorial / 2^number_of_dependencies : ℕ) = 1260 := by
  sorry

end valid_scheduling_orders_l1889_188908


namespace pens_in_drawer_l1889_188900

/-- The number of pens in Maria's desk drawer -/
def total_pens (red_pens black_pens blue_pens : ℕ) : ℕ :=
  red_pens + black_pens + blue_pens

/-- Theorem stating the total number of pens in Maria's desk drawer -/
theorem pens_in_drawer : 
  let red_pens : ℕ := 8
  let black_pens : ℕ := red_pens + 10
  let blue_pens : ℕ := red_pens + 7
  total_pens red_pens black_pens blue_pens = 41 := by
  sorry

end pens_in_drawer_l1889_188900


namespace no_negative_roots_and_positive_root_exists_l1889_188947

def f (x : ℝ) : ℝ := x^6 - 3*x^5 - 6*x^3 - x + 8

theorem no_negative_roots_and_positive_root_exists :
  (∀ x < 0, f x ≠ 0) ∧ (∃ x > 0, f x = 0) := by
  sorry

end no_negative_roots_and_positive_root_exists_l1889_188947


namespace like_terms_sum_l1889_188965

theorem like_terms_sum (a b : ℝ) (x y : ℝ) 
  (h : 3 * a^(7*x) * b^(y+7) = 5 * a^(2-4*y) * b^(2*x)) : x + y = -1 := by
  sorry

end like_terms_sum_l1889_188965


namespace inscribed_sphere_radius_bound_l1889_188986

/-- A tetrahedron with an inscribed sphere -/
structure Tetrahedron where
  /-- Length of one pair of opposite edges -/
  a : ℝ
  /-- Length of the other pair of opposite edges -/
  b : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- Ensure a and b are positive -/
  ha : 0 < a
  hb : 0 < b
  /-- Ensure r is positive -/
  hr : 0 < r

/-- The radius of the inscribed sphere is less than ab/(2(a+b)) -/
theorem inscribed_sphere_radius_bound (t : Tetrahedron) : t.r < (t.a * t.b) / (2 * (t.a + t.b)) := by
  sorry

end inscribed_sphere_radius_bound_l1889_188986


namespace two_and_one_third_of_x_is_42_l1889_188967

theorem two_and_one_third_of_x_is_42 : ∃ x : ℚ, (7/3) * x = 42 ∧ x = 18 := by
  sorry

end two_and_one_third_of_x_is_42_l1889_188967


namespace not_divisible_by_11599_l1889_188934

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def N : ℚ := (factorial 3400) / ((factorial 1700) ^ 2)

theorem not_divisible_by_11599 : ¬ (∃ (k : ℤ), N = k * (11599 : ℚ)) := by
  sorry

end not_divisible_by_11599_l1889_188934


namespace max_sum_of_three_naturals_l1889_188907

theorem max_sum_of_three_naturals (a b c : ℕ) (h1 : a + b = 1014) (h2 : c - b = 497) (h3 : a > b) :
  (∀ a' b' c' : ℕ, a' + b' = 1014 → c' - b' = 497 → a' > b' → a' + b' + c' ≤ a + b + c) ∧
  a + b + c = 2017 :=
by sorry

end max_sum_of_three_naturals_l1889_188907


namespace smallest_integer_above_sqrt_difference_power_l1889_188926

theorem smallest_integer_above_sqrt_difference_power :
  ∃ n : ℤ, (n = 9737 ∧ ∀ m : ℤ, (m > (Real.sqrt 5 - Real.sqrt 3)^8 → m ≥ n)) :=
by sorry

end smallest_integer_above_sqrt_difference_power_l1889_188926


namespace jace_neighbor_payment_l1889_188905

/-- Proves that Jace gave 0 cents to his neighbor -/
theorem jace_neighbor_payment (earned : ℕ) (debt : ℕ) (remaining : ℕ) : 
  earned = 1000 → debt = 358 → remaining = 642 → (earned - debt - remaining) * 100 = 0 := by
  sorry

end jace_neighbor_payment_l1889_188905


namespace reverse_digit_integers_l1889_188938

theorem reverse_digit_integers (q r : ℕ) : 
  (10 ≤ q ∧ q < 100) →  -- q is a two-digit integer
  (10 ≤ r ∧ r < 100) →  -- r is a two-digit integer
  (∃ a b : ℕ, q = 10 * a + b ∧ r = 10 * b + a) →  -- q and r have reversed digits
  (q > r → q - r < 60) →  -- positive difference less than 60
  (r > q → r - q < 60) →  -- positive difference less than 60
  (∀ x y : ℕ, (10 ≤ x ∧ x < 100) → (10 ≤ y ∧ y < 100) → 
    (∃ c d : ℕ, x = 10 * c + d ∧ y = 10 * d + c) → 
    (x > y → x - y ≤ 54) ∧ (y > x → y - x ≤ 54)) →  -- greatest possible difference is 54
  (∃ a b : ℕ, q = 10 * a + b ∧ r = 10 * b + a ∧ a = b + 6) :=  -- conclusion: tens digit is 6 more than units digit
by sorry

end reverse_digit_integers_l1889_188938


namespace figure_division_l1889_188957

/-- A figure consisting of 24 cells can be divided into equal parts of specific sizes. -/
theorem figure_division (n : ℕ) : n ∣ 24 ∧ n ≠ 1 ↔ n ∈ ({2, 3, 4, 6, 8, 12, 24} : Finset ℕ) :=
sorry

end figure_division_l1889_188957


namespace total_students_is_240_l1889_188935

/-- The number of students from Know It All High School -/
def know_it_all_students : ℕ := 50

/-- The number of students from Karen High School -/
def karen_students : ℕ := (3 * know_it_all_students) / 5

/-- The combined number of students from Know It All High School and Karen High School -/
def combined_students : ℕ := know_it_all_students + karen_students

/-- The number of students from Novel Corona High School -/
def novel_corona_students : ℕ := 2 * combined_students

/-- The total number of students at the competition -/
def total_students : ℕ := combined_students + novel_corona_students

/-- Theorem stating that the total number of students at the competition is 240 -/
theorem total_students_is_240 : total_students = 240 := by
  sorry

end total_students_is_240_l1889_188935


namespace banana_permutations_l1889_188994

/-- The number of distinct permutations of a sequence with repeated elements -/
def distinctPermutations (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

/-- The theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations :
  distinctPermutations 6 [3, 2, 1] = 60 := by
  sorry

#eval distinctPermutations 6 [3, 2, 1]

end banana_permutations_l1889_188994


namespace log_inequality_l1889_188904

theorem log_inequality (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  Real.log ((4 * x - 5) / |x - 2|) / Real.log (x^2) ≥ 1/2 ↔ -1 + Real.sqrt 6 ≤ x ∧ x ≤ 5 :=
by sorry

end log_inequality_l1889_188904


namespace product_of_roots_l1889_188991

theorem product_of_roots (x : ℝ) : 
  (6 = 2 * x^2 + 4 * x) → 
  (let a := 2
   let b := 4
   let c := -6
   c / a = -3) := by
sorry

end product_of_roots_l1889_188991


namespace trapezoid_diagonal_midpoint_segment_length_l1889_188956

/-- A trapezoid with upper base length L and midline length m -/
structure Trapezoid (L m : ℝ) where
  upper_base : ℝ := L
  midline : ℝ := m

/-- The length of the segment connecting the midpoints of the two diagonals in a trapezoid -/
def diagonal_midpoint_segment_length (T : Trapezoid L m) : ℝ :=
  T.midline - T.upper_base

theorem trapezoid_diagonal_midpoint_segment_length (L m : ℝ) (T : Trapezoid L m) :
  diagonal_midpoint_segment_length T = m - L := by
  sorry

end trapezoid_diagonal_midpoint_segment_length_l1889_188956


namespace seating_arrangements_l1889_188978

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The total number of children -/
def total_children : ℕ := num_boys + num_girls

/-- Calculates the number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways boys can sit together -/
def boys_together : ℕ := permutations num_boys num_boys * permutations (num_girls + 1) (num_girls + 1)

/-- The number of arrangements where no two girls sit next to each other -/
def girls_not_adjacent : ℕ := permutations num_boys num_boys * permutations (num_boys + 1) num_girls

/-- The number of ways boys can sit together and girls can sit together -/
def boys_and_girls_together : ℕ := permutations num_boys num_boys * permutations num_girls num_girls * permutations 2 2

/-- The number of arrangements where a specific boy doesn't sit at the beginning and a specific girl doesn't sit at the end -/
def specific_positions : ℕ := permutations total_children total_children - 2 * permutations (total_children - 1) (total_children - 1) + permutations (total_children - 2) (total_children - 2)

theorem seating_arrangements :
  boys_together = 576 ∧
  girls_not_adjacent = 1440 ∧
  boys_and_girls_together = 288 ∧
  specific_positions = 3720 := by sorry

end seating_arrangements_l1889_188978


namespace cube_root_of_nested_roots_l1889_188980

theorem cube_root_of_nested_roots (x : ℝ) (h : x ≥ 0) :
  (x * (x * x^(1/3))^(1/2))^(1/3) = x^(5/9) := by
  sorry

end cube_root_of_nested_roots_l1889_188980


namespace line_segment_endpoint_l1889_188954

/-- Given a line segment with midpoint (-3, 4) and one endpoint (0, 2),
    prove that the other endpoint is (-6, 6) -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) :
  midpoint = (-3, 4) →
  endpoint1 = (0, 2) →
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-6, 6) := by
  sorry

end line_segment_endpoint_l1889_188954


namespace nested_G_evaluation_l1889_188942

def G (x : ℝ) : ℝ := (x - 2)^2 - 1

theorem nested_G_evaluation : G (G (G (G (G 2)))) = 1179395 := by
  sorry

end nested_G_evaluation_l1889_188942


namespace michelle_crayons_l1889_188902

/-- The number of crayons Michelle has -/
def total_crayons (crayons_per_box : ℝ) (num_boxes : ℝ) : ℝ :=
  crayons_per_box * num_boxes

/-- Proof that Michelle has 7.0 crayons -/
theorem michelle_crayons :
  total_crayons 5.0 1.4 = 7.0 := by
  sorry

end michelle_crayons_l1889_188902


namespace triangle_angle_problem_l1889_188975

theorem triangle_angle_problem (α : Real) 
  (h1 : 0 < α ∧ α < π) -- α is an internal angle of a triangle
  (h2 : Real.sin α + Real.cos α = 1/5) :
  (Real.tan α = -4/3) ∧ 
  ((Real.sin (3*π/2 + α) * Real.sin (π/2 - α) * Real.tan (π - α)^3) / 
   (Real.cos (π/2 + α) * Real.cos (3*π/2 - α)) = -4/3) := by
sorry

end triangle_angle_problem_l1889_188975


namespace original_recipe_flour_amount_l1889_188936

/-- Given a recipe that uses 8 ounces of butter for some amount of flour,
    and knowing that 12 ounces of butter is used for 56 cups of flour
    when the recipe is quadrupled, prove that the original recipe
    requires 37 cups of flour. -/
theorem original_recipe_flour_amount :
  ∀ (x : ℚ),
  (8 : ℚ) / x = (12 : ℚ) / (4 * 56) →
  x = 37 := by
sorry

end original_recipe_flour_amount_l1889_188936


namespace cricket_average_score_l1889_188910

theorem cricket_average_score 
  (avg_2_matches : ℝ) 
  (avg_5_matches : ℝ) 
  (num_matches : ℕ) 
  (h1 : avg_2_matches = 20) 
  (h2 : avg_5_matches = 26) 
  (h3 : num_matches = 5) :
  let remaining_matches := num_matches - 2
  let total_score_5 := avg_5_matches * num_matches
  let total_score_2 := avg_2_matches * 2
  let remaining_score := total_score_5 - total_score_2
  remaining_score / remaining_matches = 30 := by
sorry

end cricket_average_score_l1889_188910


namespace binomial_distribution_unique_parameters_l1889_188945

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (b : BinomialDistribution) : ℝ := b.n * b.p

/-- The variance of a binomial distribution -/
def variance (b : BinomialDistribution) : ℝ := b.n * b.p * (1 - b.p)

/-- Theorem: For a binomial distribution X ~ B(n, p) with E(X) = 3 and D(X) = 2,
    the values of n and p are 9 and 1/3 respectively -/
theorem binomial_distribution_unique_parameters :
  ∀ b : BinomialDistribution,
    expectedValue b = 3 →
    variance b = 2 →
    b.n = 9 ∧ b.p = 1/3 := by
  sorry

end binomial_distribution_unique_parameters_l1889_188945


namespace a_gt_one_sufficient_not_necessary_for_a_sq_gt_a_l1889_188968

theorem a_gt_one_sufficient_not_necessary_for_a_sq_gt_a :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a^2 > a) := by
  sorry

end a_gt_one_sufficient_not_necessary_for_a_sq_gt_a_l1889_188968


namespace smallest_five_digit_congruent_to_2_mod_17_l1889_188983

theorem smallest_five_digit_congruent_to_2_mod_17 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 2 → n ≥ 10013 :=
by sorry

end smallest_five_digit_congruent_to_2_mod_17_l1889_188983


namespace solve_earnings_l1889_188982

def earnings_problem (first_month_daily : ℝ) : Prop :=
  let second_month_daily := 2 * first_month_daily
  let third_month_daily := second_month_daily
  let first_month_total := 30 * first_month_daily
  let second_month_total := 30 * second_month_daily
  let third_month_total := 15 * third_month_daily
  first_month_total + second_month_total + third_month_total = 1200

theorem solve_earnings : ∃ (x : ℝ), earnings_problem x ∧ x = 10 := by
  sorry

end solve_earnings_l1889_188982


namespace function_inequality_l1889_188930

theorem function_inequality (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x * Real.log x - a * x ≥ -x^2 - 2) → a ≤ -2 := by
  sorry

end function_inequality_l1889_188930


namespace x_greater_than_y_l1889_188974

theorem x_greater_than_y (x y z : ℝ) 
  (eq1 : x + y + z = 28)
  (eq2 : 2 * x - y = 32)
  (pos_x : x > 0)
  (pos_y : y > 0)
  (pos_z : z > 0) :
  x > y := by
  sorry

end x_greater_than_y_l1889_188974


namespace function_range_l1889_188964

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem function_range :
  {y | ∃ x ∈ domain, f x = y} = {y | -1 ≤ y ∧ y < 3} := by sorry

end function_range_l1889_188964


namespace randys_final_amount_l1889_188960

/-- Calculates Randy's remaining money after a series of transactions --/
def randys_remaining_money (initial_amount : ℝ) (smith_gift : ℝ) 
  (sally_percentage : ℝ) (stock_percentage : ℝ) (crypto_percentage : ℝ) : ℝ :=
  let new_total := initial_amount + smith_gift
  let after_sally := new_total * (1 - sally_percentage)
  let after_stocks := after_sally * (1 - stock_percentage)
  after_stocks * (1 - crypto_percentage)

/-- Theorem stating that Randy's remaining money is $1,008 --/
theorem randys_final_amount :
  randys_remaining_money 3000 200 0.25 0.40 0.30 = 1008 := by
  sorry

#eval randys_remaining_money 3000 200 0.25 0.40 0.30

end randys_final_amount_l1889_188960


namespace exponent_simplification_l1889_188984

theorem exponent_simplification (x : ℝ) : 4 * x^3 - 3 * x^3 = x^3 := by
  sorry

end exponent_simplification_l1889_188984


namespace quadratic_root_value_l1889_188977

theorem quadratic_root_value (d : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + d = 0 ↔ x = (3 + Real.sqrt d) / 2 ∨ x = (3 - Real.sqrt d) / 2) →
  d = 9/5 := by
sorry

end quadratic_root_value_l1889_188977


namespace largest_divisor_of_n_l1889_188972

theorem largest_divisor_of_n (n : ℕ) (h1 : 0 < n) (h2 : 72 ∣ n^2) :
  ∃ (v : ℕ), v = 12 ∧ v ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ v :=
by sorry

end largest_divisor_of_n_l1889_188972


namespace optimal_carriages_and_passengers_l1889_188943

/-- The daily round-trip frequency as a function of the number of carriages -/
def y (x : ℕ) : ℝ := -2 * x + 24

/-- The total number of carriages operated daily -/
def S (x : ℕ) : ℝ := x * y x

/-- The daily number of passengers transported -/
def W (x : ℕ) : ℝ := 110 * S x

/-- The constraint on the number of carriages -/
def valid_x (x : ℕ) : Prop := 0 ≤ x ∧ x ≤ 12

theorem optimal_carriages_and_passengers :
  ∃ (x : ℕ), valid_x x ∧
    (∀ (x' : ℕ), valid_x x' → W x' ≤ W x) ∧
    x = 6 ∧
    W x = 7920 :=
sorry

end optimal_carriages_and_passengers_l1889_188943


namespace thirtieth_term_is_61_l1889_188929

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem thirtieth_term_is_61 :
  arithmetic_sequence 3 2 30 = 61 := by
  sorry

end thirtieth_term_is_61_l1889_188929


namespace trig_problem_l1889_188922

theorem trig_problem (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.sin (α + Real.pi / 6) = 3 / 5) : 
  Real.cos (2 * α - Real.pi / 6) = 24 / 25 := by
  sorry

end trig_problem_l1889_188922


namespace xy_sum_eleven_l1889_188923

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem xy_sum_eleven (x y : ℝ) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hxy : x * y = 10) 
  (hineq : x^(log2 x) * y^(log2 y) ≥ 10) : 
  x + y = 11 := by
sorry

end xy_sum_eleven_l1889_188923


namespace equilateral_triangle_expression_bound_l1889_188950

theorem equilateral_triangle_expression_bound (a : ℝ) (h : a > 0) : (3 * a^2) / (3 * a) > 0 := by
  sorry

end equilateral_triangle_expression_bound_l1889_188950


namespace angle_D_measure_l1889_188987

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180

-- Define the quadrilateral formed by drawing a line
structure Quadrilateral (t : Triangle) where
  D : ℝ
  line_sum_180 : D + (180 - t.A - t.B) = 180

-- Theorem statement
theorem angle_D_measure (t : Triangle) (q : Quadrilateral t) 
  (h1 : t.A = 85) (h2 : t.B = 34) : q.D = 119 := by
  sorry

end angle_D_measure_l1889_188987


namespace inequality_solution_l1889_188990

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end inequality_solution_l1889_188990


namespace triangle_area_l1889_188906

/-- The area of a triangle with vertices at (0,0), (0,5), and (7,12) is 17.5 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (0, 5)
  let v3 : ℝ × ℝ := (7, 12)
  (1/2 : ℝ) * |v2.2 - v1.2| * |v3.1 - v1.1| = 17.5 := by
  sorry

end triangle_area_l1889_188906


namespace partner_a_profit_share_l1889_188921

/-- Calculates the share of profit for partner A given the initial investments,
    changes after 8 months, and total profit at the end of the year. -/
theorem partner_a_profit_share
  (a_initial : ℕ)
  (b_initial : ℕ)
  (a_change : ℤ)
  (b_change : ℕ)
  (total_profit : ℕ)
  (h1 : a_initial = 6000)
  (h2 : b_initial = 4000)
  (h3 : a_change = -1000)
  (h4 : b_change = 1000)
  (h5 : total_profit = 630) :
  ((a_initial * 8 + (a_initial + a_change) * 4) * total_profit) /
  ((a_initial * 8 + (a_initial + a_change) * 4) + (b_initial * 8 + (b_initial + b_change) * 4)) = 357 :=
by sorry

end partner_a_profit_share_l1889_188921


namespace solution_set_of_inequality_l1889_188958

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4 * x^2 - x - 5

-- Define the solution set
def S : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5/4 }

-- Theorem statement
theorem solution_set_of_inequality :
  { x : ℝ | f x ≤ 0 } = S :=
sorry

end solution_set_of_inequality_l1889_188958


namespace monthly_rate_is_42_l1889_188909

/-- The monthly parking rate that satisfies the given conditions -/
def monthly_rate : ℚ :=
  let weekly_rate : ℚ := 10
  let weeks_per_year : ℕ := 52
  let months_per_year : ℕ := 12
  let yearly_savings : ℚ := 16
  (weekly_rate * weeks_per_year - yearly_savings) / months_per_year

/-- Proof that the monthly parking rate is $42 -/
theorem monthly_rate_is_42 : monthly_rate = 42 := by
  sorry

end monthly_rate_is_42_l1889_188909


namespace geometric_sequence_ratio_l1889_188944

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  (a₁ + a₁ * q + a₁ * q^2 = 3 * a₁) → (q = -2 ∨ q = 1) :=
by sorry

end geometric_sequence_ratio_l1889_188944


namespace students_neither_football_nor_cricket_l1889_188966

theorem students_neither_football_nor_cricket 
  (total : ℕ) (football : ℕ) (cricket : ℕ) (both : ℕ) 
  (h1 : total = 410)
  (h2 : football = 325)
  (h3 : cricket = 175)
  (h4 : both = 140) :
  total - (football + cricket - both) = 50 := by
sorry

end students_neither_football_nor_cricket_l1889_188966


namespace largest_square_tile_size_l1889_188976

/-- The length of the courtyard in centimeters -/
def courtyard_length : ℕ := 378

/-- The width of the courtyard in centimeters -/
def courtyard_width : ℕ := 525

/-- The size of the largest square tile in centimeters -/
def largest_tile_size : ℕ := 21

theorem largest_square_tile_size :
  (courtyard_length % largest_tile_size = 0) ∧
  (courtyard_width % largest_tile_size = 0) ∧
  ∀ (tile_size : ℕ), tile_size > largest_tile_size →
    (courtyard_length % tile_size ≠ 0) ∨ (courtyard_width % tile_size ≠ 0) :=
by sorry

end largest_square_tile_size_l1889_188976


namespace trig_expression_equality_l1889_188948

theorem trig_expression_equality : 
  (Real.sin (38 * π / 180) * Real.sin (38 * π / 180) + 
   Real.cos (38 * π / 180) * Real.sin (52 * π / 180) - 
   Real.tan (15 * π / 180) ^ 2) / 
  (3 * Real.tan (15 * π / 180)) = 
  (2 + Real.sqrt 3) / 9 := by
  sorry

end trig_expression_equality_l1889_188948


namespace no_triangle_solution_l1889_188973

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ

-- Theorem stating that no triangle exists with the given conditions
theorem no_triangle_solution :
  ¬ ∃ (t : Triangle), t.a = 181 ∧ t.b = 209 ∧ t.A = 121 := by
  sorry

end no_triangle_solution_l1889_188973


namespace equation_solution_l1889_188961

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (18 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 34 := by
  sorry

end equation_solution_l1889_188961


namespace pet_store_birds_l1889_188915

/-- Calculates the total number of birds in a pet store given the number of cages and birds per cage. -/
def total_birds (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) : ℕ :=
  num_cages * (parrots_per_cage + parakeets_per_cage)

/-- Proves that the pet store has 72 birds in total. -/
theorem pet_store_birds : total_birds 9 2 6 = 72 := by
  sorry

end pet_store_birds_l1889_188915


namespace smallest_stairs_l1889_188911

theorem smallest_stairs (n : ℕ) : 
  n > 15 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 → 
  n ≥ 52 :=
by sorry

end smallest_stairs_l1889_188911


namespace function_characterization_l1889_188963

theorem function_characterization (f : ℚ → ℚ) 
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end function_characterization_l1889_188963


namespace largest_n_with_perfect_squares_l1889_188953

theorem largest_n_with_perfect_squares : ∃ (N : ℤ),
  (∃ (a : ℤ), N + 496 = a^2) ∧
  (∃ (b : ℤ), N + 224 = b^2) ∧
  (∀ (M : ℤ), M > N →
    ¬(∃ (x : ℤ), M + 496 = x^2) ∨
    ¬(∃ (y : ℤ), M + 224 = y^2)) ∧
  N = 4265 :=
sorry

end largest_n_with_perfect_squares_l1889_188953


namespace gizmo_production_l1889_188916

/-- Represents the production rate of gadgets per worker per hour -/
def gadget_rate : ℝ := 2

/-- Represents the production rate of gizmos per worker per hour -/
def gizmo_rate : ℝ := 1.5

/-- Represents the number of workers -/
def workers : ℕ := 40

/-- Represents the total working hours -/
def total_hours : ℝ := 6

/-- Represents the number of gadgets to be produced -/
def gadgets_to_produce : ℕ := 240

theorem gizmo_production :
  let hours_for_gadgets : ℝ := gadgets_to_produce / (workers * gadget_rate)
  let remaining_hours : ℝ := total_hours - hours_for_gadgets
  ↑workers * gizmo_rate * remaining_hours = 180 :=
sorry

end gizmo_production_l1889_188916


namespace f_unique_zero_x1_minus_2x2_bound_l1889_188946

noncomputable section

variables (a : ℝ) (h : a ≥ 0)

def f (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

def g (x : ℝ) : ℝ := a * Real.exp x + x

theorem f_unique_zero :
  ∃! x, f a x = 0 :=
sorry

theorem x1_minus_2x2_bound (x₁ x₂ : ℝ) 
  (h₁ : x₁ > -1) (h₂ : x₂ > -1) 
  (h₃ : f a x₁ = g a x₁ - g a x₂) :
  x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
sorry

end f_unique_zero_x1_minus_2x2_bound_l1889_188946


namespace michaels_fruit_cost_l1889_188940

/-- Calculates the total cost of fruit for pies -/
def total_fruit_cost (peach_pies apple_pies blueberry_pies : ℕ) 
                     (fruit_per_pie : ℕ) 
                     (apple_blueberry_price peach_price : ℚ) : ℚ :=
  let peach_pounds := peach_pies * fruit_per_pie
  let apple_pounds := apple_pies * fruit_per_pie
  let blueberry_pounds := blueberry_pies * fruit_per_pie
  let apple_blueberry_cost := (apple_pounds + blueberry_pounds) * apple_blueberry_price
  let peach_cost := peach_pounds * peach_price
  apple_blueberry_cost + peach_cost

/-- Theorem: The total cost of fruit for Michael's pie order is $51.00 -/
theorem michaels_fruit_cost :
  total_fruit_cost 5 4 3 3 1 2 = 51 := by
  sorry

end michaels_fruit_cost_l1889_188940


namespace concert_ticket_revenue_l1889_188901

/-- Calculates the total revenue from concert ticket sales --/
theorem concert_ticket_revenue :
  let full_price : ℕ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let first_discount_percent : ℕ := 40
  let second_discount_percent : ℕ := 15
  let total_attendees : ℕ := 56

  let first_group_revenue := first_group_size * (full_price * (100 - first_discount_percent) / 100)
  let second_group_revenue := second_group_size * (full_price * (100 - second_discount_percent) / 100)
  let remaining_attendees := total_attendees - first_group_size - second_group_size
  let remaining_revenue := remaining_attendees * full_price

  let total_revenue := first_group_revenue + second_group_revenue + remaining_revenue

  total_revenue = 980 := by
    sorry

end concert_ticket_revenue_l1889_188901


namespace six_right_triangles_with_smallest_perimeter_l1889_188951

/-- A structure representing a triangle with integer sides -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Check if a triangle is a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2

/-- Calculate the perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ :=
  t.a + t.b + t.c

/-- The set of six triangles with their side lengths -/
def six_triangles : List Triangle :=
  [⟨120, 288, 312⟩, ⟨144, 270, 306⟩, ⟨72, 320, 328⟩,
   ⟨45, 336, 339⟩, ⟨80, 315, 325⟩, ⟨180, 240, 300⟩]

/-- Theorem: There exist 6 rational right triangles with the same smallest possible perimeter of 720 -/
theorem six_right_triangles_with_smallest_perimeter :
  (∀ t ∈ six_triangles, is_right_triangle t) ∧
  (∀ t ∈ six_triangles, perimeter t = 720) ∧
  (∀ t : Triangle, is_right_triangle t → perimeter t < 720 → t ∉ six_triangles) :=
sorry

end six_right_triangles_with_smallest_perimeter_l1889_188951


namespace largest_angle_in_triangle_l1889_188952

theorem largest_angle_in_triangle (a b c : ℝ) (A : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + a * c + b * c = 2 * b →
  a - a * c + b * c = 2 * c →
  a = b + c + 2 * b * c * Real.cos A →
  A = 2 * π / 3 :=
by sorry

end largest_angle_in_triangle_l1889_188952


namespace tangent_line_at_one_l1889_188920

noncomputable def f (x : ℝ) := x * Real.exp x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m * x + b ↔ 
    (∃ (h : ℝ → ℝ), (∀ t, t ≠ 1 → (h t - f 1) / (t - 1) = (f t - f 1) / (t - 1)) ∧
                     (h 1 = f 1) ∧
                     y = h x) :=
by sorry

end tangent_line_at_one_l1889_188920


namespace unique_provider_choices_l1889_188985

theorem unique_provider_choices (n m : ℕ) (hn : n = 23) (hm : m = 4) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) = 213840 := by
  sorry

end unique_provider_choices_l1889_188985


namespace largest_initial_number_l1889_188993

theorem largest_initial_number : ∃ (a b c d e : ℕ), 
  189 + a + b + c + d + e = 200 ∧ 
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
  189 % a ≠ 0 ∧ 189 % b ≠ 0 ∧ 189 % c ≠ 0 ∧ 189 % d ≠ 0 ∧ 189 % e ≠ 0 ∧
  ∀ n > 189, ¬(∃ (x y z w v : ℕ), 
    n + x + y + z + w + v = 200 ∧ 
    x ≥ 2 ∧ y ≥ 2 ∧ z ≥ 2 ∧ w ≥ 2 ∧ v ≥ 2 ∧
    n % x ≠ 0 ∧ n % y ≠ 0 ∧ n % z ≠ 0 ∧ n % w ≠ 0 ∧ n % v ≠ 0) :=
by sorry

end largest_initial_number_l1889_188993


namespace min_sum_tangents_l1889_188928

/-- In an acute triangle ABC, given that a = 2b * sin(C), 
    the minimum value of tan(A) + tan(B) + tan(C) is 3√3 -/
theorem min_sum_tangents (A B C : Real) (a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a = 2 * b * Real.sin C ∧  -- Given condition
  a = c * Real.sin A ∧  -- Law of sines
  b = c * Real.sin B ∧  -- Law of sines
  c = c * Real.sin C  -- Law of sines
  →
  (Real.tan A + Real.tan B + Real.tan C ≥ 3 * (3 : Real).sqrt) ∧
  ∃ (A' B' C' : Real), 
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧
    A' + B' + C' = π ∧
    Real.tan A' + Real.tan B' + Real.tan C' = 3 * (3 : Real).sqrt :=
by sorry

end min_sum_tangents_l1889_188928


namespace triangle_area_is_168_l1889_188925

-- Define the function representing the curve
def f (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define x-intercepts
def x_intercept1 : ℝ := 4
def x_intercept2 : ℝ := -3

-- Define y-intercept
def y_intercept : ℝ := f 0

-- Theorem statement
theorem triangle_area_is_168 :
  let base : ℝ := x_intercept1 - x_intercept2
  let height : ℝ := y_intercept
  (1/2 : ℝ) * base * height = 168 := by sorry

end triangle_area_is_168_l1889_188925
