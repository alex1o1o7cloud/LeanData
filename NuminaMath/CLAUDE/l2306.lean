import Mathlib

namespace twelve_triangles_fit_l2306_230610

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  base : ℕ
  height : ℕ

/-- Calculates the area of a right triangle -/
def area (t : RightTriangle) : ℕ := t.base * t.height / 2

/-- Counts the number of small triangles that fit into a large triangle -/
def count_triangles (large : RightTriangle) (small : RightTriangle) : ℕ :=
  area large / area small

/-- Theorem stating that 12 small triangles fit into the large triangle -/
theorem twelve_triangles_fit (large small : RightTriangle) 
  (h1 : large.base = 6) (h2 : large.height = 4)
  (h3 : small.base = 2) (h4 : small.height = 1) :
  count_triangles large small = 12 := by
  sorry

end twelve_triangles_fit_l2306_230610


namespace equation_C_violates_basic_properties_l2306_230635

-- Define the equations
def equation_A (a b c : ℝ) : Prop := (a / c = b / c) → (a = b)
def equation_B (a b : ℝ) : Prop := (-a = -b) → (2 - a = 2 - b)
def equation_C (a b c : ℝ) : Prop := (a * c = b * c) → (a = b)
def equation_D (a b m : ℝ) : Prop := ((m^2 + 1) * a = (m^2 + 1) * b) → (a = b)

-- Theorem statement
theorem equation_C_violates_basic_properties :
  (∃ a b c : ℝ, ¬(equation_C a b c)) ∧
  (∀ a b c : ℝ, c ≠ 0 → equation_A a b c) ∧
  (∀ a b : ℝ, equation_B a b) ∧
  (∀ a b m : ℝ, equation_D a b m) :=
by sorry

end equation_C_violates_basic_properties_l2306_230635


namespace parabola_point_x_coordinate_l2306_230607

/-- The x-coordinate of a point on the parabola y^2 = 6x that is twice as far from the focus as from the y-axis -/
theorem parabola_point_x_coordinate 
  (x y : ℝ) 
  (h1 : y^2 = 6*x) -- Point is on the parabola y^2 = 6x
  (h2 : (x - 3/2)^2 + y^2 = 4 * x^2) -- Distance to focus is twice distance to y-axis
  : x = 3/2 := by sorry

end parabola_point_x_coordinate_l2306_230607


namespace product_equality_l2306_230622

theorem product_equality (h : 213 * 16 = 3408) : 1.6 * 213.0 = 340.8 := by
  sorry

end product_equality_l2306_230622


namespace center_radius_sum_l2306_230637

/-- Definition of the circle D -/
def D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - 14*p.1 + p.2^2 + 10*p.2 = -34}

/-- Center of the circle D -/
def center : ℝ × ℝ := sorry

/-- Radius of the circle D -/
def radius : ℝ := sorry

/-- Theorem stating the sum of center coordinates and radius -/
theorem center_radius_sum :
  center.1 + center.2 + radius = 2 + 2 * Real.sqrt 10 := by sorry

end center_radius_sum_l2306_230637


namespace complex_magnitude_l2306_230603

theorem complex_magnitude (z₁ z₂ : ℂ) 
  (h1 : z₁ + z₂ = Complex.I * z₁) 
  (h2 : z₂^2 = 2 * Complex.I) : 
  Complex.abs z₁ = 1 := by sorry

end complex_magnitude_l2306_230603


namespace symmetric_point_x_axis_l2306_230647

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricPointXAxis (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

theorem symmetric_point_x_axis :
  let original := Point3D.mk (-2) 1 4
  symmetricPointXAxis original = Point3D.mk (-2) (-1) (-4) := by
  sorry

end symmetric_point_x_axis_l2306_230647


namespace sqrt_product_simplification_l2306_230688

theorem sqrt_product_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (45 * x^2) * Real.sqrt (8 * x^3) * Real.sqrt (22 * x) = 60 * x^3 * Real.sqrt 55 := by
  sorry

end sqrt_product_simplification_l2306_230688


namespace distance_between_specific_planes_l2306_230655

/-- The distance between two planes given by their equations -/
def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ := sorry

/-- Theorem: The distance between the planes 2x - 4y + 4z = 10 and 4x - 8y + 8z = 20 is 0 -/
theorem distance_between_specific_planes :
  distance_between_planes 2 (-4) 4 10 4 (-8) 8 20 = 0 := by
  sorry

end distance_between_specific_planes_l2306_230655


namespace equation_solution_l2306_230617

theorem equation_solution :
  ∃ y : ℝ, (5 : ℝ)^(2*y) * (25 : ℝ)^y = (625 : ℝ)^3 ∧ y = 3 :=
by
  -- Define 25 and 625 in terms of 5
  have h1 : (25 : ℝ) = (5 : ℝ)^2 := by sorry
  have h2 : (625 : ℝ) = (5 : ℝ)^4 := by sorry

  -- Prove the existence of y
  sorry

end equation_solution_l2306_230617


namespace difference_of_numbers_l2306_230687

theorem difference_of_numbers (x y : ℝ) : 
  x + y = 20 → x^2 - y^2 = 160 → x - y = 8 := by sorry

end difference_of_numbers_l2306_230687


namespace base_seven_23456_equals_6068_l2306_230645

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_23456_equals_6068 :
  base_seven_to_ten [6, 5, 4, 3, 2] = 6068 := by
  sorry

end base_seven_23456_equals_6068_l2306_230645


namespace tan_1450_degrees_solution_l2306_230683

theorem tan_1450_degrees_solution (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1450 * π / 180) →
  n = 10 ∨ n = -170 := by
sorry

end tan_1450_degrees_solution_l2306_230683


namespace gcd_lcm_product_24_60_l2306_230681

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end gcd_lcm_product_24_60_l2306_230681


namespace complex_equation_solution_l2306_230699

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) (a : ℝ) : Prop :=
  z / (a + 2 * i) = i

-- Define the condition that real part equals imaginary part
def real_equals_imag (z : ℂ) : Prop :=
  z.re = z.im

-- The theorem to prove
theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : given_equation z a) 
  (h2 : real_equals_imag (z / (a + 2 * i))) : 
  a = -2 := by
  sorry

end complex_equation_solution_l2306_230699


namespace spelling_contest_problem_l2306_230624

theorem spelling_contest_problem (drew_wrong carla_correct total : ℕ) 
  (h1 : drew_wrong = 6)
  (h2 : carla_correct = 14)
  (h3 : total = 52)
  (h4 : 2 * drew_wrong = carla_correct + (total - (carla_correct + drew_wrong + (total - (2 * drew_wrong + carla_correct))))) :
  total - (2 * drew_wrong + carla_correct) = 20 := by
  sorry

end spelling_contest_problem_l2306_230624


namespace max_books_borrowed_l2306_230636

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 38) (h2 : zero_books = 2) (h3 : one_book = 12) 
  (h4 : two_books = 10) (h5 : avg_books = 2) : ∃ (max_books : ℕ), max_books = 5 ∧ 
  (∀ (student_books : ℕ), student_books ≤ max_books) := by
  sorry

end max_books_borrowed_l2306_230636


namespace cos_330_degrees_l2306_230654

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l2306_230654


namespace quadratic_root_ratio_sum_l2306_230612

theorem quadratic_root_ratio_sum (x₁ x₂ : ℝ) : 
  x₁^2 + 2*x₁ - 8 = 0 →
  x₂^2 + 2*x₂ - 8 = 0 →
  x₁ ≠ 0 →
  x₂ ≠ 0 →
  x₂/x₁ + x₁/x₂ = -5/2 := by
sorry

end quadratic_root_ratio_sum_l2306_230612


namespace unique_integral_solution_l2306_230661

theorem unique_integral_solution (x y z n : ℤ) 
  (h1 : x * y + y * z + z * x = 3 * n^2 - 1)
  (h2 : x + y + z = 3 * n)
  (h3 : x ≥ y ∧ y ≥ z) :
  x = n + 1 ∧ y = n ∧ z = n - 1 :=
by sorry

end unique_integral_solution_l2306_230661


namespace oliver_seashells_l2306_230670

/-- The number of seashells Oliver collected. -/
def total_seashells : ℕ := 4

/-- The number of seashells Oliver collected on Tuesday. -/
def tuesday_seashells : ℕ := 2

/-- The number of seashells Oliver collected on Monday. -/
def monday_seashells : ℕ := total_seashells - tuesday_seashells

theorem oliver_seashells :
  monday_seashells = total_seashells - tuesday_seashells :=
by sorry

end oliver_seashells_l2306_230670


namespace big_al_bananas_l2306_230619

theorem big_al_bananas (a : ℕ) (h : a + 2*a + 4*a + 8*a + 16*a = 155) : 16*a = 80 := by
  sorry

end big_al_bananas_l2306_230619


namespace waiter_tips_ratio_l2306_230652

theorem waiter_tips_ratio (salary tips : ℝ) 
  (h : tips / (salary + tips) = 0.6363636363636364) :
  tips / salary = 1.75 := by
sorry

end waiter_tips_ratio_l2306_230652


namespace condition_property_l2306_230632

theorem condition_property :
  (∀ x y : ℝ, x + y ≠ 5 → (x ≠ 1 ∨ y ≠ 4)) ∧
  (∃ x y : ℝ, (x ≠ 1 ∨ y ≠ 4) ∧ x + y = 5) := by
  sorry

end condition_property_l2306_230632


namespace triangle_angle_values_l2306_230684

theorem triangle_angle_values (A B C : Real) (AB AC : Real) :
  AB = 2 →
  AC = Real.sqrt 2 →
  B = 30 * Real.pi / 180 →
  A = 105 * Real.pi / 180 ∨ A = 15 * Real.pi / 180 :=
by sorry

end triangle_angle_values_l2306_230684


namespace sphere_circle_paint_equivalence_l2306_230658

theorem sphere_circle_paint_equivalence (r_sphere r_circle : ℝ) : 
  r_sphere = 3 → 
  4 * π * r_sphere^2 = π * r_circle^2 → 
  r_circle = 6 := by
  sorry

end sphere_circle_paint_equivalence_l2306_230658


namespace parabola_b_value_l2306_230695

-- Define the parabola equation
def parabola (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem parabola_b_value :
  ∀ a b : ℝ,
  (parabola a b 2 = 10) →
  (parabola a b (-2) = 6) →
  b = 4 := by
sorry

end parabola_b_value_l2306_230695


namespace positive_c_in_quadratic_with_no_roots_l2306_230651

/-- A quadratic trinomial with no roots and positive sum of coefficients has a positive constant term. -/
theorem positive_c_in_quadratic_with_no_roots 
  (a b c : ℝ) 
  (no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) 
  (sum_positive : a + b + c > 0) : 
  c > 0 := by
  sorry

end positive_c_in_quadratic_with_no_roots_l2306_230651


namespace irrational_sum_two_l2306_230602

theorem irrational_sum_two : ∃ (a b : ℝ), Irrational a ∧ Irrational b ∧ a + b = 2 := by
  sorry

end irrational_sum_two_l2306_230602


namespace book_selection_problem_l2306_230641

/-- The number of ways to choose 2 books from 15 books, excluding 3 pairs that cannot be chosen. -/
theorem book_selection_problem (total_books : Nat) (books_to_choose : Nat) (prohibited_pairs : Nat) : 
  total_books = 15 → books_to_choose = 2 → prohibited_pairs = 3 →
  Nat.choose total_books books_to_choose - prohibited_pairs = 102 := by
sorry

end book_selection_problem_l2306_230641


namespace last_twelve_average_l2306_230633

theorem last_twelve_average (total_count : Nat) (total_average : ℚ) (first_twelve_average : ℚ) (thirteenth_result : ℚ) :
  total_count = 25 →
  total_average = 24 →
  first_twelve_average = 14 →
  thirteenth_result = 228 →
  (total_count * total_average = 12 * first_twelve_average + thirteenth_result + 12 * ((total_count * total_average - 12 * first_twelve_average - thirteenth_result) / 12)) ∧
  ((total_count * total_average - 12 * first_twelve_average - thirteenth_result) / 12 = 17) := by
sorry

end last_twelve_average_l2306_230633


namespace right_triangle_cone_volume_l2306_230697

theorem right_triangle_cone_volume (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / 3 : ℝ) * π * y^2 * x = 1500 * π ∧
  (1 / 3 : ℝ) * π * x^2 * y = 540 * π →
  Real.sqrt (x^2 + y^2) = 5 * Real.sqrt 34 := by
  sorry

end right_triangle_cone_volume_l2306_230697


namespace kyle_lifting_improvement_l2306_230601

theorem kyle_lifting_improvement (current_capacity : ℕ) (ratio : ℕ) : 
  current_capacity = 80 ∧ ratio = 3 → 
  current_capacity - (current_capacity / ratio) = 53 := by
sorry

end kyle_lifting_improvement_l2306_230601


namespace days_in_year_l2306_230638

theorem days_in_year (a b c : ℕ+) (h : 29 * a + 30 * b + 31 * c = 366) : 
  19 * a + 20 * b + 21 * c = 246 := by
  sorry

end days_in_year_l2306_230638


namespace mixture_volume_l2306_230653

/-- Given a mixture of two liquids p and q with an initial ratio of 5:3,
    if adding 15 liters of liquid q changes the ratio to 5:6,
    then the initial volume of the mixture was 40 liters. -/
theorem mixture_volume (p q : ℝ) (h1 : p / q = 5 / 3) 
    (h2 : p / (q + 15) = 5 / 6) : p + q = 40 := by
  sorry

end mixture_volume_l2306_230653


namespace sports_club_size_l2306_230685

/-- The number of members in a sports club -/
def sports_club_members (badminton tennis both neither : ℕ) : ℕ :=
  badminton + tennis - both + neither

/-- Theorem: The sports club has 40 members -/
theorem sports_club_size :
  sports_club_members 20 18 3 5 = 40 := by
  sorry

end sports_club_size_l2306_230685


namespace city_population_problem_l2306_230662

theorem city_population_problem :
  ∃ (N : ℕ),
    (∃ (x : ℕ), N = x^2) ∧
    (∃ (y : ℕ), N + 100 = y^2 + 1) ∧
    (∃ (z : ℕ), N + 200 = z^2) ∧
    (∃ (k : ℕ), N = 7 * k) :=
by
  sorry

end city_population_problem_l2306_230662


namespace committee_probability_l2306_230693

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  let total_ways := Nat.choose total_members committee_size
  let unwanted_cases := Nat.choose num_girls committee_size +
                        num_boys * Nat.choose num_girls (committee_size - 1) +
                        Nat.choose num_boys committee_size +
                        num_girls * Nat.choose num_boys (committee_size - 1)
  (total_ways - unwanted_cases : ℚ) / total_ways = 457215 / 593775 := by
  sorry

end committee_probability_l2306_230693


namespace equation_solutions_function_property_l2306_230677

-- Part a
theorem equation_solutions (x : ℝ) : 2^x = x + 1 ↔ x = 0 ∨ x = 1 := by sorry

-- Part b
theorem function_property (f : ℝ → ℝ) (h : ∀ x, (f ∘ f) x = 2^x - 1) : f 0 + f 1 = 1 := by sorry

end equation_solutions_function_property_l2306_230677


namespace find_k_l2306_230644

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem find_k (k : ℤ) (h_odd : k % 2 = 1) (h_eq : f (f (f k)) = 31) : k = 119 := by
  sorry

end find_k_l2306_230644


namespace like_terms_exponent_sum_l2306_230631

theorem like_terms_exponent_sum (m n : ℤ) : 
  (∃ (x y : ℝ), -5 * x^m * y^(m+1) = x^(n-1) * y^3) → m + n = 5 := by
sorry

end like_terms_exponent_sum_l2306_230631


namespace sqrt_product_sqrt_l2306_230672

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_product_sqrt_l2306_230672


namespace quadratic_real_root_l2306_230686

theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end quadratic_real_root_l2306_230686


namespace sum_of_exponents_of_sqrt_largest_perfect_square_15_factorial_l2306_230682

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define a function to calculate the exponent of a prime factor in n!
def primeExponentInFactorial (n : ℕ) (p : ℕ) : ℕ :=
  if p.Prime then
    (List.range (n + 1)).foldl (λ acc k => acc + k / p^k) 0
  else
    0

-- Define a function to get the largest even number not exceeding n
def largestEvenNotExceeding (n : ℕ) : ℕ :=
  if n % 2 = 0 then n else n - 1

-- Define the main theorem
theorem sum_of_exponents_of_sqrt_largest_perfect_square_15_factorial :
  (let n := 15
   let primes := [2, 3, 5, 7]
   let exponents := primes.map (λ p => largestEvenNotExceeding (primeExponentInFactorial n p) / 2)
   exponents.sum) = 10 := by
  sorry

end sum_of_exponents_of_sqrt_largest_perfect_square_15_factorial_l2306_230682


namespace volleyball_team_starters_l2306_230608

theorem volleyball_team_starters (n : ℕ) (q : ℕ) (s : ℕ) (h1 : n = 16) (h2 : q = 4) (h3 : s = 6) :
  (Nat.choose (n - q) s) + q * (Nat.choose (n - q) (s - 1)) = 4092 :=
by sorry

end volleyball_team_starters_l2306_230608


namespace boxwood_count_proof_l2306_230614

/-- The cost to trim up each boxwood -/
def trim_cost : ℚ := 5

/-- The cost to trim a boxwood into a fancy shape -/
def fancy_trim_cost : ℚ := 15

/-- The number of boxwoods to be shaped into spheres -/
def fancy_trim_count : ℕ := 4

/-- The total charge for the service -/
def total_charge : ℚ := 210

/-- The number of boxwood hedges the customer wants trimmed up -/
def boxwood_count : ℕ := 30

theorem boxwood_count_proof :
  trim_cost * boxwood_count + fancy_trim_cost * fancy_trim_count = total_charge :=
by sorry

end boxwood_count_proof_l2306_230614


namespace required_run_rate_is_6_15_l2306_230611

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  targetScore : ℕ
  firstSegmentOvers : ℕ
  firstSegmentRunRate : ℚ
  firstSegmentWicketsLost : ℕ
  maxTotalWicketsLost : ℕ
  personalMilestone : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstSegmentOvers
  let runsScored := game.firstSegmentRunRate * game.firstSegmentOvers
  let runsNeeded := game.targetScore - runsScored
  runsNeeded / remainingOvers

/-- Theorem stating the required run rate for the given game scenario -/
theorem required_run_rate_is_6_15 (game : CricketGame) 
    (h1 : game.totalOvers = 50)
    (h2 : game.targetScore = 282)
    (h3 : game.firstSegmentOvers = 10)
    (h4 : game.firstSegmentRunRate = 3.6)
    (h5 : game.firstSegmentWicketsLost = 2)
    (h6 : game.maxTotalWicketsLost = 5)
    (h7 : game.personalMilestone = 75) :
    requiredRunRate game = 6.15 := by
  sorry


end required_run_rate_is_6_15_l2306_230611


namespace no_real_solutions_l2306_230642

theorem no_real_solutions : ¬∃ (x y : ℝ), 3*x^2 + y^2 - 9*x - 6*y + 23 = 0 := by
  sorry

end no_real_solutions_l2306_230642


namespace pizza_payment_difference_l2306_230613

theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let pepperoni_slices : ℕ := total_slices / 3
  let plain_cost : ℚ := 12
  let pepperoni_cost : ℚ := 3
  let total_cost : ℚ := plain_cost + pepperoni_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let pepperoni_slice_cost : ℚ := cost_per_slice + pepperoni_cost / pepperoni_slices
  let plain_slice_cost : ℚ := cost_per_slice
  let mark_pepperoni_slices : ℕ := pepperoni_slices
  let mark_plain_slices : ℕ := 2
  let anne_slices : ℕ := total_slices - mark_pepperoni_slices - mark_plain_slices
  let mark_cost : ℚ := mark_pepperoni_slices * pepperoni_slice_cost + mark_plain_slices * plain_slice_cost
  let anne_cost : ℚ := anne_slices * plain_slice_cost
  mark_cost - anne_cost = 3 := by sorry

end pizza_payment_difference_l2306_230613


namespace village_population_decrease_rate_l2306_230625

/-- Proves that the rate of decrease in Village X's population is 1,200 people per year -/
theorem village_population_decrease_rate 
  (initial_x : ℕ) 
  (initial_y : ℕ) 
  (growth_rate_y : ℕ) 
  (years : ℕ) 
  (h1 : initial_x = 70000)
  (h2 : initial_y = 42000)
  (h3 : growth_rate_y = 800)
  (h4 : years = 14)
  (h5 : ∃ (decrease_rate : ℕ), initial_x - years * decrease_rate = initial_y + years * growth_rate_y) :
  ∃ (decrease_rate : ℕ), decrease_rate = 1200 := by
sorry

end village_population_decrease_rate_l2306_230625


namespace purchase_costs_l2306_230615

def cost (x y : ℕ) : ℕ := x + 2 * y

theorem purchase_costs : 
  (cost 5 5 ≤ 18) ∧ 
  (cost 9 4 ≤ 18) ∧ 
  (cost 9 5 > 18) ∧ 
  (cost 2 6 ≤ 18) ∧ 
  (cost 16 0 ≤ 18) :=
by sorry

end purchase_costs_l2306_230615


namespace max_value_rational_function_l2306_230663

theorem max_value_rational_function (x : ℝ) (h : x < -1) :
  (x^2 + 7*x + 10) / (x + 1) ≤ 1 ∧
  (x^2 + 7*x + 10) / (x + 1) = 1 ↔ x = -3 :=
by sorry

end max_value_rational_function_l2306_230663


namespace sum_of_fractions_l2306_230640

theorem sum_of_fractions : (1 : ℚ) / 3 + 2 / 7 + 3 / 8 = 167 / 168 := by
  sorry

end sum_of_fractions_l2306_230640


namespace eventually_constant_l2306_230648

/-- S(n) is defined as n - m^2, where m is the greatest integer with m^2 ≤ n -/
def S (n : ℕ) : ℕ :=
  n - (Nat.sqrt n) ^ 2

/-- The sequence a_k is defined recursively -/
def a (A : ℕ) : ℕ → ℕ
  | 0 => A
  | k + 1 => a A k + S (a A k)

/-- The main theorem stating the condition for the sequence to be eventually constant -/
theorem eventually_constant (A : ℕ) :
  (∃ k : ℕ, ∀ n ≥ k, a A n = a A k) ↔ ∃ m : ℕ, A = m ^ 2 := by
  sorry

end eventually_constant_l2306_230648


namespace no_valid_assignment_l2306_230689

/-- Represents a vertex of the hexagon or its center -/
inductive Vertex
| A | B | C | D | E | F | G

/-- Represents a triangle formed by the center and two adjacent vertices -/
structure Triangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- The set of all triangles in the hexagon -/
def hexagonTriangles : List Triangle := [
  ⟨Vertex.A, Vertex.B, Vertex.G⟩,
  ⟨Vertex.B, Vertex.C, Vertex.G⟩,
  ⟨Vertex.C, Vertex.D, Vertex.G⟩,
  ⟨Vertex.D, Vertex.E, Vertex.G⟩,
  ⟨Vertex.E, Vertex.F, Vertex.G⟩,
  ⟨Vertex.F, Vertex.A, Vertex.G⟩
]

/-- A function that assigns an integer to each vertex -/
def VertexAssignment := Vertex → Int

/-- Checks if the integers assigned to a triangle are in ascending order clockwise -/
def isAscendingClockwise (assignment : VertexAssignment) (t : Triangle) : Prop :=
  assignment t.v1 < assignment t.v2 ∧ assignment t.v2 < assignment t.v3

/-- The main theorem stating that no valid assignment exists -/
theorem no_valid_assignment :
  ¬∃ (assignment : VertexAssignment),
    (∀ v1 v2 : Vertex, v1 ≠ v2 → assignment v1 ≠ assignment v2) ∧
    (∀ t ∈ hexagonTriangles, isAscendingClockwise assignment t) :=
sorry


end no_valid_assignment_l2306_230689


namespace tangent_points_and_circle_area_l2306_230643

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Define the tangent points M and N
def M (x₁ : ℝ) : ℝ × ℝ := (x₁, parabola x₁)
def N (x₂ : ℝ) : ℝ × ℝ := (x₂, parabola x₂)

-- State the theorem
theorem tangent_points_and_circle_area 
  (x₁ x₂ : ℝ) 
  (h_tangent : ∃ (k b : ℝ), ∀ x, k * x + b = parabola x → x = x₁ ∨ x = x₂)
  (h_order : x₁ < x₂) :
  (x₁ = 1 - Real.sqrt 2 ∧ x₂ = 1 + Real.sqrt 2) ∧ 
  (∃ (r : ℝ), r > 0 ∧ 
    (∃ (x y : ℝ), (x - P.1)^2 + (y - P.2)^2 = r^2 ∧
      ∃ (k b : ℝ), k * x + b = y ∧ k * x₁ + b = parabola x₁ ∧ k * x₂ + b = parabola x₂) ∧
    π * r^2 = 16 * π / 5) := by
  sorry

end tangent_points_and_circle_area_l2306_230643


namespace count_goats_l2306_230692

/-- Given a field with animals, prove the number of goats -/
theorem count_goats (total : ℕ) (cows : ℕ) (sheep_and_goats : ℕ) 
  (h1 : total = 200)
  (h2 : cows = 40)
  (h3 : sheep_and_goats = 56)
  : total - cows - sheep_and_goats = 104 := by
  sorry

end count_goats_l2306_230692


namespace laundry_day_lcm_l2306_230604

theorem laundry_day_lcm : Nat.lcm 6 9 = 18 := by
  sorry

end laundry_day_lcm_l2306_230604


namespace factorization_problem_1_l2306_230627

theorem factorization_problem_1 (x : ℝ) :
  x^4 - 8*x^2 + 4 = (x^2 + 2*x - 2) * (x^2 - 2*x - 2) := by
sorry

end factorization_problem_1_l2306_230627


namespace staircase_theorem_l2306_230628

def staircase_problem (first_staircase : ℕ) (step_height : ℚ) : ℚ :=
  let second_staircase := 2 * first_staircase
  let third_staircase := second_staircase - 10
  let total_steps := first_staircase + second_staircase + third_staircase
  total_steps * step_height

theorem staircase_theorem :
  staircase_problem 20 (1/2) = 45 := by
  sorry

end staircase_theorem_l2306_230628


namespace combine_like_terms_1_combine_like_terms_2_l2306_230680

-- Problem 1
theorem combine_like_terms_1 (a : ℝ) :
  2*a^2 - 3*a - 5 + 4*a + a^2 = 3*a^2 + a - 5 := by sorry

-- Problem 2
theorem combine_like_terms_2 (m n : ℝ) :
  2*m^2 + 5/2*n^2 - 1/3*(m^2 - 6*n^2) = 5/3*m^2 + 9/2*n^2 := by sorry

end combine_like_terms_1_combine_like_terms_2_l2306_230680


namespace leo_current_weight_l2306_230650

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 92

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 160 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 160

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) ∧
  (leo_weight = 92) := by
sorry

end leo_current_weight_l2306_230650


namespace sara_salad_cost_l2306_230616

/-- The cost of Sara's lunch items -/
structure LunchCost where
  hotdog : ℝ
  total : ℝ

/-- Calculates the cost of the salad given the total lunch cost and hotdog cost -/
def salad_cost (lunch : LunchCost) : ℝ :=
  lunch.total - lunch.hotdog

/-- Theorem stating that Sara's salad cost $5.10 -/
theorem sara_salad_cost :
  let lunch : LunchCost := { hotdog := 5.36, total := 10.46 }
  salad_cost lunch = 5.10 := by
  sorry

end sara_salad_cost_l2306_230616


namespace system_solution_l2306_230698

theorem system_solution (a b : ℝ) : 
  (∃ x y : ℝ, x + y = a ∧ 2 * x + y = 16 ∧ x = 6 ∧ y = b) → 
  a = 10 ∧ b = 4 := by
  sorry

end system_solution_l2306_230698


namespace not_in_range_iff_b_in_interval_l2306_230671

-- Define the function f
def f (b x : ℝ) : ℝ := x^2 + b*x + 2

-- Theorem statement
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, f b x ≠ -2) ↔ b ∈ Set.Ioo (-4 : ℝ) 4 := by
  sorry

end not_in_range_iff_b_in_interval_l2306_230671


namespace half_abs_diff_squares_21_17_l2306_230666

theorem half_abs_diff_squares_21_17 : (1/2 : ℚ) * |21^2 - 17^2| = 76 := by
  sorry

end half_abs_diff_squares_21_17_l2306_230666


namespace frog_corner_probability_l2306_230657

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents a direction of hop -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- The grid on which the frog hops -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a position is a corner -/
def isCorner (p : Position) : Bool :=
  (p.x = 0 ∧ p.y = 0) ∨ (p.x = 0 ∧ p.y = 3) ∨ (p.x = 3 ∧ p.y = 0) ∨ (p.x = 3 ∧ p.y = 3)

/-- Performs a single hop in the given direction with wrap-around -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up    => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down  => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Direction.Left  => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- Calculates the probability of reaching a corner within n hops -/
def probReachCorner (start : Position) (n : Nat) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem frog_corner_probability :
  probReachCorner ⟨1, 1⟩ 5 = 15/16 :=
sorry

end frog_corner_probability_l2306_230657


namespace workshop_workers_l2306_230694

/-- Represents the total number of workers in a workshop -/
def total_workers : ℕ := 20

/-- Represents the number of technicians -/
def technicians : ℕ := 5

/-- Represents the average salary of all workers -/
def avg_salary_all : ℕ := 750

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 900

/-- Represents the average salary of non-technician workers -/
def avg_salary_others : ℕ := 700

/-- Theorem stating that given the conditions, the total number of workers is 20 -/
theorem workshop_workers : 
  (total_workers * avg_salary_all = technicians * avg_salary_technicians + 
   (total_workers - technicians) * avg_salary_others) → 
  total_workers = 20 :=
by sorry

end workshop_workers_l2306_230694


namespace count_eight_digit_numbers_with_product_4900_l2306_230691

/-- The number of eight-digit numbers whose digits' product equals 4900 -/
def eight_digit_numbers_with_product_4900 : ℕ := 4200

/-- Theorem stating that the number of eight-digit numbers whose digits' product equals 4900 is 4200 -/
theorem count_eight_digit_numbers_with_product_4900 :
  eight_digit_numbers_with_product_4900 = 4200 := by
  sorry

end count_eight_digit_numbers_with_product_4900_l2306_230691


namespace fraction_reciprocal_product_l2306_230673

theorem fraction_reciprocal_product : (1 / (5 / 3)) * (5 / 3) = 1 := by sorry

end fraction_reciprocal_product_l2306_230673


namespace quadratic_real_roots_condition_l2306_230646

/-- 
For a quadratic equation (a-1)x^2 - 2x + 1 = 0 to have real roots, 
a must satisfy: a ≤ 2 and a ≠ 1 
-/
theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2 ∧ a ≠ 1) :=
by sorry

end quadratic_real_roots_condition_l2306_230646


namespace jelly_bean_theorem_l2306_230620

def jelly_bean_problem (total_jelly_beans : ℕ) (total_people : ℕ) (last_four_take : ℕ) (remaining : ℕ) : Prop :=
  let last_four_total := 4 * last_four_take
  let taken_by_others := total_jelly_beans - remaining - last_four_total
  let others_take_each := 2 * last_four_take
  let num_others := taken_by_others / others_take_each
  num_others = 6 ∧ 
  num_others + 4 = total_people ∧
  taken_by_others + last_four_total + remaining = total_jelly_beans

theorem jelly_bean_theorem : jelly_bean_problem 8000 10 400 1600 := by
  sorry

end jelly_bean_theorem_l2306_230620


namespace complex_modulus_problem_l2306_230667

theorem complex_modulus_problem (z : ℂ) : z * Complex.I ^ 2018 = 3 + 4 * Complex.I → Complex.abs z = 5 := by
  sorry

end complex_modulus_problem_l2306_230667


namespace apple_bags_theorem_l2306_230676

def is_valid_total (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 6 * a + 12 * b ∧ 70 ≤ n ∧ n ≤ 80

theorem apple_bags_theorem : 
  {n : ℕ | is_valid_total n} = {72, 78} :=
sorry

end apple_bags_theorem_l2306_230676


namespace school_fee_calculation_l2306_230664

/-- Represents the number of bills of each denomination given by a parent -/
structure BillCount where
  fifty : Nat
  twenty : Nat
  ten : Nat

/-- Calculates the total value of bills given by a parent -/
def totalValue (bills : BillCount) : Nat :=
  50 * bills.fifty + 20 * bills.twenty + 10 * bills.ten

theorem school_fee_calculation (mother father : BillCount)
    (h_mother : mother = { fifty := 1, twenty := 2, ten := 3 })
    (h_father : father = { fifty := 4, twenty := 1, ten := 1 }) :
    totalValue mother + totalValue father = 350 := by
  sorry

end school_fee_calculation_l2306_230664


namespace smallest_solution_congruence_l2306_230659

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l2306_230659


namespace derivative_at_negative_one_l2306_230668

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem derivative_at_negative_one 
  (a b c : ℝ) 
  (h : (4 * a + 2 * b) = 2) : 
  (4 * a * (-1)^3 + 2 * b * (-1)) = -2 := by sorry

end derivative_at_negative_one_l2306_230668


namespace complex_number_in_fourth_quadrant_l2306_230656

/-- The complex number z = (2 + 3i) / (1 + 2i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 + 3*I) / (1 + 2*I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l2306_230656


namespace hyperbola_eccentricity_l2306_230679

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity : ∀ (a b : ℝ) (P : ℝ × ℝ),
  a > 0 → b > 0 →
  -- Hyperbola equation
  P.1^2 / a^2 - P.2^2 / b^2 = 1 →
  -- P is on the curve y = √x
  P.2 = Real.sqrt P.1 →
  -- Tangent line passes through the left focus (-1, 0)
  (Real.sqrt P.1 - 0) / (P.1 - (-1)) = 1 / (2 * Real.sqrt P.1) →
  -- The eccentricity is (√5 + 1) / 2
  a / Real.sqrt (a^2 + b^2) = (Real.sqrt 5 + 1) / 2 := by
  sorry

end hyperbola_eccentricity_l2306_230679


namespace distinct_integers_sum_l2306_230639

theorem distinct_integers_sum (b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℕ) : 
  b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧ b₂ ≠ b₇ ∧ b₂ ≠ b₈ ∧ b₂ ≠ b₉ ∧
  b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧ b₃ ≠ b₇ ∧ b₃ ≠ b₈ ∧ b₃ ≠ b₉ ∧
  b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧ b₄ ≠ b₇ ∧ b₄ ≠ b₈ ∧ b₄ ≠ b₉ ∧
  b₅ ≠ b₆ ∧ b₅ ≠ b₇ ∧ b₅ ≠ b₈ ∧ b₅ ≠ b₉ ∧
  b₆ ≠ b₇ ∧ b₆ ≠ b₈ ∧ b₆ ≠ b₉ ∧
  b₇ ≠ b₈ ∧ b₇ ≠ b₉ ∧
  b₈ ≠ b₉ →
  (7 : ℚ) / 11 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 + b₉ / 362880 →
  0 ≤ b₂ ∧ b₂ < 2 →
  0 ≤ b₃ ∧ b₃ < 3 →
  0 ≤ b₄ ∧ b₄ < 4 →
  0 ≤ b₅ ∧ b₅ < 5 →
  0 ≤ b₆ ∧ b₆ < 6 →
  0 ≤ b₇ ∧ b₇ < 7 →
  0 ≤ b₈ ∧ b₈ < 8 →
  0 ≤ b₉ ∧ b₉ < 9 →
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ + b₈ + b₉ = 16 := by
sorry

end distinct_integers_sum_l2306_230639


namespace group_size_problem_l2306_230605

theorem group_size_problem (x : ℕ) : 
  (5 * x + 45 = 7 * x + 3) → x = 21 := by
  sorry

end group_size_problem_l2306_230605


namespace largest_constant_inequality_two_is_largest_constant_l2306_230675

theorem largest_constant_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 :=
sorry

theorem two_is_largest_constant :
  ∀ ε > 0, ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
    Real.sqrt (e / (a + b + c + d)) < 2 + ε :=
sorry

end largest_constant_inequality_two_is_largest_constant_l2306_230675


namespace locus_of_P_perpendicular_line_through_focus_l2306_230600

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point M on the ellipse
def point_M (x y : ℝ) : Prop := ellipse_C x y

-- Define the point N as the foot of the perpendicular from M to x-axis
def point_N (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the point P
def point_P (x y : ℝ) (mx my : ℝ) : Prop :=
  point_M mx my ∧ (x - mx)^2 + y^2 = 2 * my^2

-- Define the point Q
def point_Q (y : ℝ) : ℝ × ℝ := (-3, y)

-- Theorem 1: The locus of P is a circle
theorem locus_of_P (x y : ℝ) :
  (∃ mx my, point_P x y mx my) → x^2 + y^2 = 2 :=
sorry

-- Theorem 2: Line through P perpendicular to OQ passes through left focus
theorem perpendicular_line_through_focus (x y qy : ℝ) (mx my : ℝ) :
  point_P x y mx my →
  (x * (-3 - x) + y * (qy - y) = 1) →
  (∃ t : ℝ, x + t * (qy - y) = -1 ∧ y - t * (-3 - x) = 0) :=
sorry

end locus_of_P_perpendicular_line_through_focus_l2306_230600


namespace average_weight_increase_l2306_230630

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 97 →
  (new_weight - old_weight) / initial_count = 4 :=
by sorry

end average_weight_increase_l2306_230630


namespace total_cost_calculation_l2306_230665

def rental_cost : ℝ := 150
def gas_needed : ℝ := 8
def gas_price : ℝ := 3.50
def mileage_expense : ℝ := 0.50
def distance_driven : ℝ := 320

theorem total_cost_calculation :
  rental_cost + gas_needed * gas_price + distance_driven * mileage_expense = 338 := by
  sorry

end total_cost_calculation_l2306_230665


namespace lattice_polygon_extension_l2306_230623

/-- A point with integer coordinates -/
def LatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

/-- A polygon with all vertices being lattice points -/
def LatticePolygon (vertices : List (ℝ × ℝ)) : Prop :=
  ∀ v ∈ vertices, LatticePoint v

/-- A convex polygon -/
def ConvexPolygon (vertices : List (ℝ × ℝ)) : Prop :=
  sorry  -- Definition of convex polygon

/-- Theorem: For any convex lattice polygon, there exists another convex lattice polygon
    that contains it and has exactly one additional vertex -/
theorem lattice_polygon_extension
  (Γ : List (ℝ × ℝ))
  (h_lattice : LatticePolygon Γ)
  (h_convex : ConvexPolygon Γ) :
  ∃ (Γ' : List (ℝ × ℝ)),
    LatticePolygon Γ' ∧
    ConvexPolygon Γ' ∧
    (∀ v ∈ Γ, v ∈ Γ') ∧
    (∃! v, v ∈ Γ' ∧ v ∉ Γ) :=
  sorry

end lattice_polygon_extension_l2306_230623


namespace shopping_tax_free_cost_l2306_230634

/-- Given a shopping trip with a total spend, sales tax paid, and tax rate,
    calculate the cost of tax-free items. -/
theorem shopping_tax_free_cost
  (total_spend : ℚ)
  (sales_tax : ℚ)
  (tax_rate : ℚ)
  (h1 : total_spend = 40)
  (h2 : sales_tax = 3/10)
  (h3 : tax_rate = 6/100)
  : ∃ (tax_free_cost : ℚ), tax_free_cost = 35 :=
by
  sorry


end shopping_tax_free_cost_l2306_230634


namespace employee_salary_problem_l2306_230674

/-- Proves that given the conditions of the problem, employee N's salary is $265 per week -/
theorem employee_salary_problem (total_salary m_salary n_salary : ℝ) : 
  total_salary = 583 →
  m_salary = 1.2 * n_salary →
  total_salary = m_salary + n_salary →
  n_salary = 265 := by
  sorry

end employee_salary_problem_l2306_230674


namespace outfit_problem_l2306_230660

/-- The number of possible outfits given shirts, pants, and restrictions -/
def num_outfits (shirts : ℕ) (pants : ℕ) (restricted_shirts : ℕ) (restricted_pants : ℕ) : ℕ :=
  (shirts - restricted_shirts) * pants + restricted_shirts * (pants - restricted_pants)

/-- Theorem stating the number of outfits for the given problem -/
theorem outfit_problem :
  num_outfits 5 4 2 1 = 18 := by
  sorry


end outfit_problem_l2306_230660


namespace chord_length_concentric_circles_l2306_230696

/-- Given two concentric circles with radii a and b (a > b), 
    if the area of the ring between them is 12½π square inches,
    then the length of a chord of the larger circle tangent to the smaller circle is 5√2 inches. -/
theorem chord_length_concentric_circles (a b : ℝ) (h1 : a > b) 
  (h2 : π * a^2 - π * b^2 = 25/2 * π) : 
  ∃ (c : ℝ), c^2 = 50 ∧ c = (2 * a^2 - 2 * b^2).sqrt := by
  sorry

end chord_length_concentric_circles_l2306_230696


namespace square_triangle_perimeter_ratio_l2306_230609

theorem square_triangle_perimeter_ratio (s_square s_triangle : ℝ) 
  (h_positive_square : s_square > 0)
  (h_positive_triangle : s_triangle > 0)
  (h_equal_perimeter : 4 * s_square = 3 * s_triangle) :
  s_triangle / s_square = 4 / 3 := by
sorry

end square_triangle_perimeter_ratio_l2306_230609


namespace train_seats_problem_l2306_230669

theorem train_seats_problem (total_cars : ℕ) 
  (half_free : ℕ) (third_free : ℕ) (all_occupied : ℕ)
  (h1 : total_cars = 18)
  (h2 : half_free + third_free + all_occupied = total_cars)
  (h3 : (half_free * 6 + third_free * 4) * 2 = total_cars * 4) :
  all_occupied = 13 := by
  sorry

end train_seats_problem_l2306_230669


namespace third_class_duration_l2306_230629

/-- Calculates the duration of the third class in a course --/
theorem third_class_duration 
  (weeks : ℕ) 
  (fixed_class_hours : ℕ) 
  (fixed_classes_per_week : ℕ) 
  (homework_hours : ℕ) 
  (total_hours : ℕ) 
  (h1 : weeks = 24)
  (h2 : fixed_class_hours = 3)
  (h3 : fixed_classes_per_week = 2)
  (h4 : homework_hours = 4)
  (h5 : total_hours = 336) :
  ∃ (third_class_hours : ℕ), 
    (fixed_classes_per_week * fixed_class_hours + third_class_hours + homework_hours) * weeks = total_hours ∧
    third_class_hours = 4 :=
by sorry

end third_class_duration_l2306_230629


namespace solve_equation_l2306_230626

theorem solve_equation (x : ℝ) : x^6 = 3^12 → x = 9 := by
  sorry

end solve_equation_l2306_230626


namespace purely_imaginary_complex_number_l2306_230649

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + ((m^2 - 3*m) : ℝ)*I) → m = 2 := by
  sorry

end purely_imaginary_complex_number_l2306_230649


namespace range_of_a_l2306_230690

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x + 2| ≤ 3) → -5 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l2306_230690


namespace selection_schemes_l2306_230678

theorem selection_schemes (num_boys num_girls : ℕ) (h1 : num_boys = 4) (h2 : num_girls = 2) :
  (num_boys : ℕ) * (num_girls : ℕ) = 8 := by
  sorry

end selection_schemes_l2306_230678


namespace lateral_face_base_angle_l2306_230606

/-- A regular quadrilateral pyramid with specific properties -/
structure RegularQuadPyramid where
  /-- The angle between a lateral edge and the base plane -/
  edge_base_angle : ℝ
  /-- The angle at the apex of the pyramid -/
  apex_angle : ℝ
  /-- The condition that edge_base_angle equals apex_angle -/
  edge_base_eq_apex : edge_base_angle = apex_angle

/-- The theorem stating the angle between the lateral face and the base plane -/
theorem lateral_face_base_angle (p : RegularQuadPyramid) :
  Real.arctan (Real.sqrt (1 + Real.sqrt 5)) =
    Real.arctan (Real.sqrt (1 + Real.sqrt 5)) :=
by sorry

end lateral_face_base_angle_l2306_230606


namespace train_length_l2306_230618

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 150 → time = 12 → ∃ length : ℝ, abs (length - 500.04) < 0.01 := by
  sorry

#check train_length

end train_length_l2306_230618


namespace quadratic_equation_solution_l2306_230621

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 4 ∧ 
  (x₁^2 - 6*x₁ + 8 = 0) ∧ (x₂^2 - 6*x₂ + 8 = 0) := by
  sorry

end quadratic_equation_solution_l2306_230621
