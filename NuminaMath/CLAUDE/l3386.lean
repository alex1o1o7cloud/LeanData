import Mathlib

namespace cut_pentagon_area_l3386_338692

/-- Represents a pentagon created by cutting a triangular corner from a rectangular sheet. -/
structure CutPentagon where
  sides : Finset ℕ
  area : ℕ

/-- The theorem stating that a pentagon with specific side lengths has an area of 770. -/
theorem cut_pentagon_area : ∃ (p : CutPentagon), p.sides = {14, 21, 22, 28, 35} ∧ p.area = 770 := by
  sorry

end cut_pentagon_area_l3386_338692


namespace composite_square_area_l3386_338642

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square composed of rectangles -/
structure CompositeSquare where
  rectangle : Rectangle
  
/-- The perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- The side length of the composite square -/
def CompositeSquare.sideLength (s : CompositeSquare) : ℝ := s.rectangle.length + s.rectangle.width

/-- The area of the composite square -/
def CompositeSquare.area (s : CompositeSquare) : ℝ := (s.sideLength) ^ 2

theorem composite_square_area (s : CompositeSquare) 
  (h : s.rectangle.perimeter = 40) : s.area = 400 := by
  sorry

end composite_square_area_l3386_338642


namespace pool_paint_area_calculation_l3386_338677

/-- Calculates the total area to be painted in a cuboid-shaped pool -/
def poolPaintArea (length width depth : ℝ) : ℝ :=
  2 * (length * depth + width * depth) + length * width

theorem pool_paint_area_calculation :
  let length : ℝ := 20
  let width : ℝ := 12
  let depth : ℝ := 2
  poolPaintArea length width depth = 368 := by
  sorry

end pool_paint_area_calculation_l3386_338677


namespace remainder_theorem_l3386_338636

theorem remainder_theorem : (7 * 10^23 + 3^25) % 11 = 5 := by
  sorry

end remainder_theorem_l3386_338636


namespace quadratic_shift_sum_l3386_338673

/-- Given a quadratic function f(x) = 2x^2 - x + 7, when shifted 6 units to the right,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 62 -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 2 * x^2 - x + 7) →
  (∀ x, g x = f (x - 6)) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 62 := by
  sorry

end quadratic_shift_sum_l3386_338673


namespace sugar_profit_problem_l3386_338661

/-- Proves the quantity of sugar sold at 18% profit given the conditions -/
theorem sugar_profit_problem (total_sugar : ℝ) (profit_rate_1 profit_rate_2 overall_profit : ℝ) 
  (h1 : total_sugar = 1000)
  (h2 : profit_rate_1 = 0.08)
  (h3 : profit_rate_2 = 0.18)
  (h4 : overall_profit = 0.14)
  (h5 : ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_sugar ∧ 
    profit_rate_1 * x + profit_rate_2 * (total_sugar - x) = overall_profit * total_sugar) :
  ∃ y : ℝ, y = 600 ∧ y = total_sugar - 
    Classical.choose (h5 : ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_sugar ∧ 
      profit_rate_1 * x + profit_rate_2 * (total_sugar - x) = overall_profit * total_sugar) :=
by sorry

end sugar_profit_problem_l3386_338661


namespace functional_equation_solution_l3386_338689

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) : 
  (∀ x : ℝ, f x = x^2) ∨ (∀ x : ℝ, f x = 0) := by
  sorry

end functional_equation_solution_l3386_338689


namespace joes_steakhouse_wages_l3386_338615

/-- Proves that the manager's hourly wage is $6.50 given the conditions from Joe's Steakhouse --/
theorem joes_steakhouse_wages (manager_wage dishwasher_wage chef_wage : ℝ) :
  chef_wage = dishwasher_wage + 0.2 * dishwasher_wage →
  dishwasher_wage = 0.5 * manager_wage →
  chef_wage = manager_wage - 2.6 →
  manager_wage = 6.5 := by
sorry

end joes_steakhouse_wages_l3386_338615


namespace heartsuit_ratio_l3386_338676

-- Define the ♡ operation
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 := by
  sorry

end heartsuit_ratio_l3386_338676


namespace triangle_area_from_lines_l3386_338630

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_from_lines :
  let line1 : ℝ → ℝ := λ x => 3 * x - 4
  let line2 : ℝ → ℝ := λ x => -2 * x + 16
  let y_axis : ℝ → ℝ := λ x => 0
  let intersection_x : ℝ := 4
  let intersection_y : ℝ := line1 intersection_x
  let y_intercept1 : ℝ := line1 0
  let y_intercept2 : ℝ := line2 0
  let base : ℝ := y_intercept2 - y_intercept1
  let height : ℝ := intersection_x
  let area : ℝ := (1 / 2) * base * height
  area = 40 := by sorry

end triangle_area_from_lines_l3386_338630


namespace crazy_silly_school_books_to_read_l3386_338652

/-- The number of books still to read in a series -/
def books_to_read (total_books read_books : ℕ) : ℕ :=
  total_books - read_books

/-- Theorem: For the 'crazy silly school' series, the number of books still to read is 10 -/
theorem crazy_silly_school_books_to_read :
  books_to_read 22 12 = 10 := by
  sorry

end crazy_silly_school_books_to_read_l3386_338652


namespace geometric_relations_l3386_338624

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define specific planes and lines
variable (α β : Plane)
variable (a b : Line)

-- State the theorem
theorem geometric_relations :
  (subset b α ∧ ¬subset a α →
    (∀ x y, parallel_lines x y → parallel_line_plane x α) ∧
    ¬(∀ x y, parallel_line_plane x α → parallel_lines x y)) ∧
  (subset a α ∧ subset b α →
    ¬(parallel α β ↔ (parallel α β ∧ parallel_line_plane b β))) :=
sorry

end geometric_relations_l3386_338624


namespace prob_king_queen_is_16_2862_l3386_338629

/-- Represents a standard deck of cards with Jokers -/
structure Deck :=
  (total_cards : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)
  (num_jokers : ℕ)

/-- The probability of drawing a King then a Queen from the deck -/
def prob_king_then_queen (d : Deck) : ℚ :=
  (d.num_kings * d.num_queens : ℚ) / ((d.total_cards * (d.total_cards - 1)) : ℚ)

/-- Our specific deck -/
def our_deck : Deck :=
  { total_cards := 54
  , num_kings := 4
  , num_queens := 4
  , num_jokers := 2 }

theorem prob_king_queen_is_16_2862 :
  prob_king_then_queen our_deck = 16 / 2862 := by
  sorry

end prob_king_queen_is_16_2862_l3386_338629


namespace v_closed_under_cube_l3386_338633

def v : Set ℕ := {n : ℕ | ∃ m : ℕ, n = m^4}

theorem v_closed_under_cube (x : ℕ) (hx : x ∈ v) : x^3 ∈ v := by
  sorry

end v_closed_under_cube_l3386_338633


namespace bruno_pen_units_l3386_338688

/-- Given that Bruno buys 2.5 units of pens and ends up with 30 pens in total,
    prove that the unit he is using is 12 pens per unit. -/
theorem bruno_pen_units (units : ℝ) (total_pens : ℕ) :
  units = 2.5 ∧ total_pens = 30 → (total_pens : ℝ) / units = 12 := by
  sorry

end bruno_pen_units_l3386_338688


namespace gcd_factorial_eight_and_factorial_six_squared_l3386_338619

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 7200 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l3386_338619


namespace polynomial_sign_intervals_l3386_338606

theorem polynomial_sign_intervals (x : ℝ) :
  x > 0 → ((x - 1) * (x - 2) * (x - 3) < 0 ↔ (x > 0 ∧ x < 1) ∨ (x > 2 ∧ x < 3)) :=
by sorry

end polynomial_sign_intervals_l3386_338606


namespace line_through_point_with_specific_intercept_ratio_l3386_338660

/-- A line passing through the point (-5,2) with an x-intercept twice its y-intercept 
    has the equation 2x + 5y = 0 or x + 2y + 1 = 0 -/
theorem line_through_point_with_specific_intercept_ratio :
  ∀ (a b c : ℝ),
    (a ≠ 0 ∨ b ≠ 0) →
    (a * (-5) + b * 2 + c = 0) →
    (∃ t : ℝ, a * (-2*t) + c = 0 ∧ b * t + c = 0) →
    ((∃ k : ℝ, a = 2*k ∧ b = 5*k ∧ c = 0) ∨ (∃ k : ℝ, a = k ∧ b = 2*k ∧ c = -k)) :=
by sorry

end line_through_point_with_specific_intercept_ratio_l3386_338660


namespace composition_of_even_is_even_l3386_338645

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end composition_of_even_is_even_l3386_338645


namespace binomial_expansion_ratio_l3386_338634

theorem binomial_expansion_ratio (n : ℕ) (a b c : ℝ) :
  n ≥ 3 →
  (∀ x : ℝ, (x + 2)^n = x^n + a * x^3 + b * x^2 + c * x + 2^n) →
  a / b = 3 / 2 →
  n = 11 := by
  sorry

end binomial_expansion_ratio_l3386_338634


namespace sin_two_phi_l3386_338621

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_two_phi_l3386_338621


namespace race_outcomes_l3386_338664

theorem race_outcomes (n : ℕ) (k : ℕ) (h : n = 6 ∧ k = 4) :
  Nat.factorial n / Nat.factorial (n - k) = 360 := by
  sorry

end race_outcomes_l3386_338664


namespace ear_muffs_proof_l3386_338638

/-- The number of ear muffs bought before December -/
def ear_muffs_before_december : ℕ := 7790 - 6444

/-- The total number of ear muffs bought -/
def total_ear_muffs : ℕ := 7790

/-- The number of ear muffs bought during December -/
def ear_muffs_during_december : ℕ := 6444

theorem ear_muffs_proof :
  ear_muffs_before_december = 1346 ∧
  total_ear_muffs = ear_muffs_before_december + ear_muffs_during_december :=
by sorry

end ear_muffs_proof_l3386_338638


namespace unique_positive_solution_l3386_338678

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) :=
by
  -- The proof goes here
  sorry

end unique_positive_solution_l3386_338678


namespace quadratic_function_extrema_l3386_338671

def f (x : ℝ) := 3 * x^2 + 6 * x - 5

theorem quadratic_function_extrema :
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x₂ = max) ∧
    min = -8 ∧ max = 19 :=
by sorry

end quadratic_function_extrema_l3386_338671


namespace divide_and_add_problem_l3386_338603

theorem divide_and_add_problem (x : ℝ) : (48 / x) + 7 = 15 → x = 6 := by
  sorry

end divide_and_add_problem_l3386_338603


namespace triangle_problem_l3386_338618

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  b * Real.cos A + (1/2) * a = c →
  (B = π/3 ∧
   (c = 5 → b = 7 → a = 8 ∧ (1/2) * a * c * Real.sin B = 10 * Real.sqrt 3) ∧
   (c = 5 → C = π/4 → a = (5 * Real.sqrt 3 + 5)/2 ∧ 
    (1/2) * a * c * Real.sin B = (75 + 25 * Real.sqrt 3)/8)) :=
by sorry

end triangle_problem_l3386_338618


namespace two_machine_completion_time_l3386_338647

theorem two_machine_completion_time (t₁ t_combined : ℝ) (x : ℝ) 
  (h₁ : t₁ > 0) (h₂ : t_combined > 0) (h₃ : x > 0) 
  (h₄ : t₁ = 6) (h₅ : t_combined = 1.5) :
  (1 / t₁ + 1 / x = 1 / t_combined) ↔ 
  (1 / 6 + 1 / x = 1 / 1.5) :=
sorry

end two_machine_completion_time_l3386_338647


namespace vacation_pictures_l3386_338668

/-- The number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := 15

/-- The number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- The number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- The number of pictures Megan still has from her vacation -/
def remaining_pictures : ℕ := zoo_pictures + museum_pictures - deleted_pictures

theorem vacation_pictures : remaining_pictures = 2 := by
  sorry

end vacation_pictures_l3386_338668


namespace estimate_students_in_range_l3386_338666

/-- Given a histogram of student heights with two adjacent rectangles, 
    estimate the number of students in the combined range. -/
theorem estimate_students_in_range 
  (total_students : ℕ) 
  (rectangle_width : ℝ) 
  (height_a : ℝ) 
  (height_b : ℝ) 
  (h_total : total_students = 1500)
  (h_width : rectangle_width = 5) :
  (rectangle_width * height_a + rectangle_width * height_b) * total_students = 
    7500 * (height_a + height_b) := by
  sorry

end estimate_students_in_range_l3386_338666


namespace probability_four_blue_l3386_338626

/-- The number of blue marbles initially in the bag -/
def initial_blue : ℕ := 10

/-- The number of red marbles initially in the bag -/
def initial_red : ℕ := 5

/-- The total number of draws -/
def total_draws : ℕ := 10

/-- The number of blue marbles we want to draw -/
def target_blue : ℕ := 4

/-- The probability of drawing a blue marble, approximated as constant throughout the process -/
def p_blue : ℚ := 2/3

/-- The probability of drawing a red marble, approximated as constant throughout the process -/
def p_red : ℚ := 1/3

/-- The probability of drawing exactly 4 blue marbles out of 10 draws -/
theorem probability_four_blue : 
  (Nat.choose total_draws target_blue : ℚ) * p_blue^target_blue * p_red^(total_draws - target_blue) = 
  (210 * 16 : ℚ) / (81 * 729) := by sorry

end probability_four_blue_l3386_338626


namespace parallelogram_altitude_base_ratio_l3386_338600

/-- For a parallelogram with area 162 sq m and base 9 m, the ratio of altitude to base is 2/1 -/
theorem parallelogram_altitude_base_ratio :
  ∀ (area base altitude : ℝ),
    area = 162 →
    base = 9 →
    area = base * altitude →
    altitude / base = 2 := by
  sorry

end parallelogram_altitude_base_ratio_l3386_338600


namespace greatest_integer_c_for_all_real_domain_l3386_338695

theorem greatest_integer_c_for_all_real_domain : 
  (∃ c : ℤ, (∀ x : ℝ, x^2 + c * x + 10 ≠ 0) ∧ 
   (∀ c' : ℤ, c' > c → ∃ x : ℝ, x^2 + c' * x + 10 = 0)) → 
  (∃ c : ℤ, c = 6 ∧ (∀ x : ℝ, x^2 + c * x + 10 ≠ 0) ∧ 
   (∀ c' : ℤ, c' > c → ∃ x : ℝ, x^2 + c' * x + 10 = 0)) :=
by sorry

end greatest_integer_c_for_all_real_domain_l3386_338695


namespace quadratic_equation_solution_l3386_338607

/-- Given a quadratic equation with coefficients a, b, and c, returns true if it has exactly one solution -/
def has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

theorem quadratic_equation_solution (b : ℝ) :
  has_one_solution 3 15 b →
  b + 3 = 36 →
  b > 3 →
  b = 33 := by
sorry

end quadratic_equation_solution_l3386_338607


namespace inscribed_squares_ratio_l3386_338669

theorem inscribed_squares_ratio : 
  ∀ x y : ℝ,
  (x > 0) →
  (y > 0) →
  (x^2 + x * 5 = 5 * 12) →
  (8/5 * y^2 + y^2 + 3/5 * y^2 = 10 * 2) →
  x / y = 96 / 85 := by
sorry

end inscribed_squares_ratio_l3386_338669


namespace quadratic_function_properties_l3386_338656

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Theorem statement
theorem quadratic_function_properties :
  (∃ (a : ℝ), f x = a * (x + 1)^2 + 4) ∧ -- Vertex form with vertex at (-1, 4)
  f 2 = -5 := by -- Passes through (2, -5)
sorry

end quadratic_function_properties_l3386_338656


namespace picture_book_shelves_l3386_338698

theorem picture_book_shelves (total_books : ℕ) (mystery_shelves : ℕ) (books_per_shelf : ℕ) :
  total_books = 32 →
  mystery_shelves = 5 →
  books_per_shelf = 4 →
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 3 :=
by sorry

end picture_book_shelves_l3386_338698


namespace quadruple_count_l3386_338654

/-- The number of ordered quadruples of positive even integers that sum to 104 -/
def n : ℕ := sorry

/-- Predicate for a quadruple of positive even integers -/
def is_valid_quadruple (x₁ x₂ x₃ x₄ : ℕ) : Prop :=
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
  Even x₁ ∧ Even x₂ ∧ Even x₃ ∧ Even x₄ ∧
  x₁ + x₂ + x₃ + x₄ = 104

/-- The main theorem stating that n/100 equals 208.25 -/
theorem quadruple_count : (n : ℚ) / 100 = 208.25 := by sorry

end quadruple_count_l3386_338654


namespace smallest_solution_absolute_value_equation_l3386_338667

theorem smallest_solution_absolute_value_equation :
  let f : ℝ → ℝ := λ x => x * |x| - 3 * x + 2
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y ∧ x = (-3 - Real.sqrt 17) / 2 :=
sorry

end smallest_solution_absolute_value_equation_l3386_338667


namespace machine_worked_two_minutes_l3386_338617

/-- Calculates the working time of a machine given its production rate and total output -/
def machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) : ℚ :=
  (total_shirts : ℚ) / (shirts_per_minute : ℚ)

/-- Proves that a machine making 3 shirts per minute that made 6 shirts worked for 2 minutes -/
theorem machine_worked_two_minutes :
  machine_working_time 3 6 = 2 := by sorry

end machine_worked_two_minutes_l3386_338617


namespace gas_fill_friday_l3386_338605

/-- Calculates the number of liters of gas Mr. Deane will fill on Friday given the conditions of the problem. -/
theorem gas_fill_friday 
  (today_liters : ℝ) 
  (today_price : ℝ) 
  (price_rollback : ℝ) 
  (total_cost : ℝ) 
  (total_liters : ℝ) 
  (h1 : today_liters = 10)
  (h2 : today_price = 1.4)
  (h3 : price_rollback = 0.4)
  (h4 : total_cost = 39)
  (h5 : total_liters = 35) :
  total_liters - today_liters = 25 := by
sorry

end gas_fill_friday_l3386_338605


namespace sqrt_k_squared_minus_pk_integer_l3386_338639

theorem sqrt_k_squared_minus_pk_integer (p : ℕ) (hp : Prime p) :
  ∀ k : ℤ, (∃ n : ℕ+, (k^2 - p * k : ℤ) = n^2) ↔ 
    (p ≠ 2 ∧ (k = ((p + 1) / 2)^2 ∨ k = -((p - 1) / 2)^2)) ∨ 
    (p = 2 ∧ False) := by
  sorry

#check sqrt_k_squared_minus_pk_integer

end sqrt_k_squared_minus_pk_integer_l3386_338639


namespace product_mod_23_l3386_338632

theorem product_mod_23 : (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 9 := by
  sorry

end product_mod_23_l3386_338632


namespace inscribed_squares_ratio_l3386_338631

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 6 ∧ b = 8 ∧ c = 10

/-- A square inscribed in the triangle with side along leg of length 6 -/
def inscribed_square_x (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x / t.a = (t.b - x) / t.c

/-- A square inscribed in the triangle with side along leg of length 8 -/
def inscribed_square_y (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.b ∧ y / t.b = (t.a - y) / t.c

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ) 
  (hx : inscribed_square_x t x) (hy : inscribed_square_y t y) : 
  x / y = 9 / 8 := by
  sorry

end inscribed_squares_ratio_l3386_338631


namespace max_equal_quotient_remainder_l3386_338641

theorem max_equal_quotient_remainder (A B C : ℕ) (h1 : A = 7 * B + C) (h2 : B = C) :
  B ≤ 6 :=
sorry

end max_equal_quotient_remainder_l3386_338641


namespace complex_real_condition_l3386_338699

theorem complex_real_condition (a : ℝ) : 
  (Complex.mk (1 / (a + 5)) (a^2 + 2*a - 15)).im = 0 → a = 3 := by
  sorry

end complex_real_condition_l3386_338699


namespace greatest_three_digit_multiple_of_17_l3386_338608

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  17 ∣ n ∧
  ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n :=
by sorry

end greatest_three_digit_multiple_of_17_l3386_338608


namespace star_equality_implies_x_equals_6_l3386_338674

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

/-- Theorem stating that if (3,3) ★ (0,0) = (x,y) ★ (3,2), then x = 6 -/
theorem star_equality_implies_x_equals_6 (x y : ℤ) :
  star 3 3 0 0 = star x y 3 2 → x = 6 := by
  sorry

end star_equality_implies_x_equals_6_l3386_338674


namespace complex_fraction_equality_l3386_338655

theorem complex_fraction_equality : 
  let x : ℂ := (1 + Complex.I * Real.sqrt 3) / 3
  1 / (x^2 + x) = 9/76 - (45 * Complex.I * Real.sqrt 3)/76 := by
  sorry

end complex_fraction_equality_l3386_338655


namespace log_equation_solution_l3386_338628

theorem log_equation_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  p * (q + 1) = q →
  (Real.log p + Real.log q = Real.log (p + q) ↔ p = q / (q + 1)) := by
sorry

end log_equation_solution_l3386_338628


namespace pizza_piece_cost_l3386_338658

/-- Given that Luigi bought 4 pizzas for $80 and each pizza was cut into 5 pieces,
    prove that the cost of each pizza piece is $4. -/
theorem pizza_piece_cost (total_pizzas : ℕ) (total_cost : ℚ) (pieces_per_pizza : ℕ) :
  total_pizzas = 4 →
  total_cost = 80 →
  pieces_per_pizza = 5 →
  total_cost / (total_pizzas * pieces_per_pizza : ℚ) = 4 := by
  sorry

end pizza_piece_cost_l3386_338658


namespace frisbee_price_problem_l3386_338640

/-- Represents the price and quantity of frisbees sold at that price -/
structure FrisbeeGroup where
  price : ℝ
  quantity : ℕ

/-- Calculates the total revenue from a group of frisbees -/
def revenue (group : FrisbeeGroup) : ℝ :=
  group.price * group.quantity

theorem frisbee_price_problem (total_frisbees : ℕ) (total_revenue : ℝ) 
    (cheap_frisbees : FrisbeeGroup) (expensive_frisbees : FrisbeeGroup) : 
    total_frisbees = 60 →
    cheap_frisbees.price = 4 →
    cheap_frisbees.quantity ≥ 20 →
    cheap_frisbees.quantity + expensive_frisbees.quantity = total_frisbees →
    revenue cheap_frisbees + revenue expensive_frisbees = total_revenue →
    total_revenue = 200 →
    expensive_frisbees.price = 3 := by
  sorry

end frisbee_price_problem_l3386_338640


namespace coin_collection_value_johns_collection_value_l3386_338665

/-- Proves the value of a coin collection given certain conditions -/
theorem coin_collection_value
  (total_coins : ℕ)
  (sample_coins : ℕ)
  (sample_value : ℚ)
  (h1 : total_coins = 24)
  (h2 : sample_coins = 8)
  (h3 : sample_value = 20)
  : ℚ
:=
by
  -- The value of the entire collection
  sorry

/-- The main theorem stating the value of John's coin collection -/
theorem johns_collection_value : coin_collection_value 24 8 20 rfl rfl rfl = 60 := by sorry

end coin_collection_value_johns_collection_value_l3386_338665


namespace right_triangle_hypotenuse_l3386_338694

theorem right_triangle_hypotenuse (x y z : ℝ) : 
  x > 0 → 
  y > 0 → 
  z > 0 → 
  y = 3 * x - 3 → 
  (1 / 2) * x * y = 72 → 
  x^2 + y^2 = z^2 → 
  z = Real.sqrt 505 := by
  sorry

end right_triangle_hypotenuse_l3386_338694


namespace a_spending_percentage_l3386_338696

/-- Proves that A spends 95% of his salary given the conditions of the problem -/
theorem a_spending_percentage 
  (total_salary : ℝ) 
  (a_salary : ℝ) 
  (b_spending_percentage : ℝ) 
  (h1 : total_salary = 7000)
  (h2 : a_salary = 5250)
  (h3 : b_spending_percentage = 0.85)
  (h4 : ∃ (a_spending_percentage : ℝ), 
    a_salary * (1 - a_spending_percentage) = (total_salary - a_salary) * (1 - b_spending_percentage)) :
  ∃ (a_spending_percentage : ℝ), a_spending_percentage = 0.95 := by
sorry

end a_spending_percentage_l3386_338696


namespace additional_money_needed_l3386_338685

def water_bottles : ℕ := 5 * 12
def original_price : ℚ := 2
def reduced_price : ℚ := 185 / 100

theorem additional_money_needed :
  water_bottles * original_price - water_bottles * reduced_price = 9 := by
  sorry

end additional_money_needed_l3386_338685


namespace gcd_x_y_eq_25_l3386_338679

/-- The sum of all even integers between 13 and 63 (inclusive) -/
def x : ℕ := (14 + 62) * 25 / 2

/-- The count of even integers between 13 and 63 (inclusive) -/
def y : ℕ := 25

/-- Theorem stating that the greatest common divisor of x and y is 25 -/
theorem gcd_x_y_eq_25 : Nat.gcd x y = 25 := by sorry

end gcd_x_y_eq_25_l3386_338679


namespace train_passing_bridge_time_l3386_338651

/-- The time it takes for a train to pass a bridge -/
theorem train_passing_bridge_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) :
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let time := total_distance / train_speed_ms
  train_length = 250 ∧ bridge_length = 150 ∧ train_speed_kmh = 35 →
  ∃ ε > 0, |time - 41.1528| < ε :=
by
  sorry


end train_passing_bridge_time_l3386_338651


namespace system_solution_l3386_338690

theorem system_solution :
  ∀ x y : ℂ,
  (x^2 + y^2 = x*y ∧ x + y = x*y) ↔
  ((x = 0 ∧ y = 0) ∨
   (x = (3 + Complex.I * Real.sqrt 3) / 2 ∧ y = (3 - Complex.I * Real.sqrt 3) / 2) ∨
   (x = (3 - Complex.I * Real.sqrt 3) / 2 ∧ y = (3 + Complex.I * Real.sqrt 3) / 2)) :=
by sorry

end system_solution_l3386_338690


namespace binomial_coefficient_ratio_l3386_338662

theorem binomial_coefficient_ratio (a b : ℕ) : 
  (a = Nat.choose 6 3) → 
  (b = Nat.choose 6 4 * 2^4) → 
  (∀ k, 0 ≤ k ∧ k ≤ 6 → Nat.choose 6 k ≤ a) → 
  (∀ k, 0 ≤ k ∧ k ≤ 6 → Nat.choose 6 k * 2^k ≤ b) → 
  b / a = 12 := by
sorry

end binomial_coefficient_ratio_l3386_338662


namespace arrangement_count_l3386_338614

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of schools -/
def num_schools : ℕ := 3

/-- Represents the total number of arrangements without restrictions -/
def total_arrangements : ℕ := (num_students.choose 2) * num_schools.factorial

/-- Represents the number of arrangements where A and B are in the same school -/
def arrangements_ab_together : ℕ := num_schools.factorial

/-- Represents the number of valid arrangements -/
def valid_arrangements : ℕ := total_arrangements - arrangements_ab_together

theorem arrangement_count : valid_arrangements = 30 := by sorry

end arrangement_count_l3386_338614


namespace expression_simplification_and_evaluation_l3386_338680

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 - 1
  ((1 / (x + 1) - 1) / (x / (x^2 - 1))) = 2 - Real.sqrt 3 := by
  sorry

end expression_simplification_and_evaluation_l3386_338680


namespace shortest_side_in_triangle_l3386_338622

/-- Given a triangle with side lengths a, b, and c, if a^2 + b^2 > 5c^2, then c is the length of the shortest side. -/
theorem shortest_side_in_triangle (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_inequality : a^2 + b^2 > 5*c^2) : 
  c ≤ a ∧ c ≤ b :=
sorry

end shortest_side_in_triangle_l3386_338622


namespace merchant_printers_l3386_338657

/-- Calculates the number of printers bought given the total cost, cost per item, and number of keyboards --/
def calculate_printers (total_cost : ℕ) (keyboard_cost : ℕ) (printer_cost : ℕ) (num_keyboards : ℕ) : ℕ :=
  (total_cost - keyboard_cost * num_keyboards) / printer_cost

theorem merchant_printers :
  calculate_printers 2050 20 70 15 = 25 := by
  sorry

end merchant_printers_l3386_338657


namespace expedition_time_theorem_l3386_338609

/-- Represents the expedition parameters and calculates the minimum time to circle the mountain. -/
def minimum_expedition_time (total_distance : ℝ) (walking_speed : ℝ) (food_capacity : ℝ) : ℝ :=
  23.5

/-- Theorem stating that the minimum time to circle the mountain under given conditions is 23.5 days. -/
theorem expedition_time_theorem (total_distance : ℝ) (walking_speed : ℝ) (food_capacity : ℝ) 
  (h1 : total_distance = 100)
  (h2 : walking_speed = 20)
  (h3 : food_capacity = 2) :
  minimum_expedition_time total_distance walking_speed food_capacity = 23.5 := by
  sorry

end expedition_time_theorem_l3386_338609


namespace intersection_A_B_when_a_neg_one_complement_A_intersect_B_empty_iff_a_gt_three_l3386_338646

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (a - x)}
def B : Set ℝ := {x | x^2 - x - 6 = 0}

-- Part 1
theorem intersection_A_B_when_a_neg_one :
  A (-1) ∩ B = {-2} := by sorry

-- Part 2
theorem complement_A_intersect_B_empty_iff_a_gt_three (a : ℝ) :
  (Set.univ \ A a) ∩ B = ∅ ↔ a > 3 := by sorry

end intersection_A_B_when_a_neg_one_complement_A_intersect_B_empty_iff_a_gt_three_l3386_338646


namespace triangle_condition_implies_isosceles_right_l3386_338653

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  |t.c^2 - t.a^2 - t.b^2| + (t.a - t.b)^2 = 0

-- Define an isosceles right triangle
def isIsoscelesRightTriangle (t : Triangle) : Prop :=
  t.a = t.b ∧ t.a^2 + t.b^2 = t.c^2

-- The theorem to be proved
theorem triangle_condition_implies_isosceles_right (t : Triangle) 
  (h : satisfiesCondition t) : isIsoscelesRightTriangle t :=
sorry

end triangle_condition_implies_isosceles_right_l3386_338653


namespace sum_base4_equals_l3386_338681

/-- Converts a base 4 number (represented as a list of digits) to a natural number. -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Converts a natural number to its base 4 representation (as a list of digits). -/
def natToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 4) ((m % 4) :: acc)
    go n []

/-- The theorem to be proved -/
theorem sum_base4_equals : 
  natToBase4 (base4ToNat [3, 0, 2] + base4ToNat [2, 2, 1] + 
              base4ToNat [1, 3, 2] + base4ToNat [0, 1, 1]) = [3, 3, 2, 2] := by
  sorry


end sum_base4_equals_l3386_338681


namespace xyz_minimum_l3386_338620

theorem xyz_minimum (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 2) :
  ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 2 → x*y*z ≤ a*b*c := by
  sorry

end xyz_minimum_l3386_338620


namespace min_quotient_value_l3386_338684

theorem min_quotient_value (a b : ℝ) 
  (ha : 100 ≤ a ∧ a ≤ 300)
  (hb : 400 ≤ b ∧ b ≤ 800)
  (hab : a + b ≤ 950) :
  (∀ a' b', 100 ≤ a' ∧ a' ≤ 300 → 400 ≤ b' ∧ b' ≤ 800 → a' + b' ≤ 950 → a / b ≤ a' / b') →
  a / b = 1 / 8 :=
by sorry

end min_quotient_value_l3386_338684


namespace volunteer_selection_count_l3386_338610

/-- The number of ways to select 3 volunteers from 5 boys and 2 girls, with at least 1 girl selected -/
def select_volunteers (num_boys : ℕ) (num_girls : ℕ) (total_selected : ℕ) : ℕ :=
  Nat.choose num_girls 1 * Nat.choose num_boys 2 +
  Nat.choose num_girls 2 * Nat.choose num_boys 1

/-- Theorem stating that the number of ways to select 3 volunteers from 5 boys and 2 girls, 
    with at least 1 girl selected, is equal to 25 -/
theorem volunteer_selection_count :
  select_volunteers 5 2 3 = 25 := by
  sorry

end volunteer_selection_count_l3386_338610


namespace line_equation_l3386_338616

/-- A line passing through the point (-2, 5) with slope -3/4 has the equation 3x + 4y - 14 = 0. -/
theorem line_equation (x y : ℝ) : 
  (∃ (L : Set (ℝ × ℝ)), 
    ((-2, 5) ∈ L) ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ L → (x₂, y₂) ∈ L → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = -3/4) ∧
    ((x, y) ∈ L ↔ 3*x + 4*y - 14 = 0)) :=
by
  sorry

end line_equation_l3386_338616


namespace polynomial_factorization_l3386_338659

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end polynomial_factorization_l3386_338659


namespace last_digit_of_power_l3386_338648

theorem last_digit_of_power (a b : ℕ) (ha : a = 954950230952380948328708) (hb : b = 470128749397540235934750230) :
  (a^b) % 10 = 4 := by
  sorry

end last_digit_of_power_l3386_338648


namespace even_function_implies_a_eq_neg_one_l3386_338697

def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 1

theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = -1 := by
  sorry

end even_function_implies_a_eq_neg_one_l3386_338697


namespace quadratic_roots_transformation_l3386_338675

/-- Given that u and v are roots of 2x^2 + 5x + 3 = 0, prove that x^2 - x + 6 = 0 has roots 2u + 3 and 2v + 3 -/
theorem quadratic_roots_transformation (u v : ℝ) :
  (2 * u^2 + 5 * u + 3 = 0) →
  (2 * v^2 + 5 * v + 3 = 0) →
  ∀ x : ℝ, (x^2 - x + 6 = 0) ↔ (x = 2*u + 3 ∨ x = 2*v + 3) :=
by sorry

end quadratic_roots_transformation_l3386_338675


namespace sophies_daily_oranges_l3386_338627

/-- The number of oranges Sophie's mom gives her every day -/
def sophies_oranges : ℕ := 20

/-- The number of grapes Hannah eats per day -/
def hannahs_grapes : ℕ := 40

/-- The number of days in the observation period -/
def observation_days : ℕ := 30

/-- The total number of fruits eaten by Sophie and Hannah during the observation period -/
def total_fruits : ℕ := 1800

/-- Theorem stating that Sophie's mom gives her 20 oranges per day -/
theorem sophies_daily_oranges :
  sophies_oranges * observation_days + hannahs_grapes * observation_days = total_fruits :=
by sorry

end sophies_daily_oranges_l3386_338627


namespace local_minimum_at_one_l3386_338635

-- Define the function f
def f (x m : ℝ) : ℝ := x * (x - m)^2

-- State the theorem
theorem local_minimum_at_one (m : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f x m ≥ f 1 m) → m = 1 := by
  sorry

end local_minimum_at_one_l3386_338635


namespace fraction_equality_l3386_338613

theorem fraction_equality : 
  (14/10 : ℚ) = 7/5 ∧ 
  (1 + 2/5 : ℚ) = 7/5 ∧ 
  (1 + 7/35 : ℚ) ≠ 7/5 ∧ 
  (1 + 4/20 : ℚ) ≠ 7/5 ∧ 
  (1 + 3/15 : ℚ) ≠ 7/5 :=
by sorry

end fraction_equality_l3386_338613


namespace min_value_of_squares_l3386_338643

theorem min_value_of_squares (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a - c = 5) :
  (∀ x y : ℝ, a > x ∧ x > y ∧ y > c → (a - x)^2 + (x - y)^2 ≥ (a - b)^2 + (b - c)^2) ∧
  (a - b)^2 + (b - c)^2 = 25/2 := by
  sorry

end min_value_of_squares_l3386_338643


namespace expression_simplification_and_evaluation_l3386_338683

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) ∧
  (0 + 2) / (0 - 2) = -1 := by
  sorry

end expression_simplification_and_evaluation_l3386_338683


namespace unique_prime_pair_with_square_differences_l3386_338602

theorem unique_prime_pair_with_square_differences : 
  ∃! (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    ∃ (a b : ℕ), a^2 = p - q ∧ b^2 = p*q - q :=
by
  sorry

end unique_prime_pair_with_square_differences_l3386_338602


namespace simplify_xy_squared_l3386_338687

theorem simplify_xy_squared (x y : ℝ) : 5 * x * y^2 - 6 * x * y^2 = -x * y^2 := by
  sorry

end simplify_xy_squared_l3386_338687


namespace system_solution_l3386_338672

def equation1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y = -12
def equation2 (x z : ℝ) : Prop := x^2 + z^2 - 6*x - 2*z = -5
def equation3 (y z : ℝ) : Prop := y^2 + z^2 - 8*y - 2*z = -7

def is_solution (x y z : ℝ) : Prop :=
  equation1 x y ∧ equation2 x z ∧ equation3 y z

theorem system_solution :
  (∀ x y z : ℝ, is_solution x y z ↔
    ((x = 1 ∧ y = 1 ∧ z = 0) ∨
     (x = 1 ∧ y = 1 ∧ z = 2) ∨
     (x = 1 ∧ y = 7 ∧ z = 0) ∨
     (x = 1 ∧ y = 7 ∧ z = 2) ∨
     (x = 5 ∧ y = 1 ∧ z = 0) ∨
     (x = 5 ∧ y = 1 ∧ z = 2) ∨
     (x = 5 ∧ y = 7 ∧ z = 0) ∨
     (x = 5 ∧ y = 7 ∧ z = 2))) :=
by sorry

end system_solution_l3386_338672


namespace binomial_equation_unique_solution_l3386_338601

theorem binomial_equation_unique_solution :
  ∃! n : ℕ, (Nat.choose 15 n + Nat.choose 15 7 = Nat.choose 16 8) ∧ n = 8 := by
  sorry

end binomial_equation_unique_solution_l3386_338601


namespace pickle_barrel_problem_l3386_338604

theorem pickle_barrel_problem (B M T G S : ℚ) : 
  M + T + G + S = B →
  B - M / 2 = B / 10 →
  B - T / 2 = B / 8 →
  B - G / 2 = B / 4 →
  B - S / 2 = B / 40 := by
sorry

end pickle_barrel_problem_l3386_338604


namespace max_value_implies_a_range_l3386_338686

def f (a x : ℝ) : ℝ := -x^2 - 2*a*x

theorem max_value_implies_a_range (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ a^2) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f a x = a^2) →
  -1 ≤ a ∧ a ≤ 0 := by
sorry

end max_value_implies_a_range_l3386_338686


namespace min_difference_of_sine_bounds_l3386_338611

open Real

theorem min_difference_of_sine_bounds (a b : ℝ) :
  (∀ x ∈ Set.Ioo 0 (π / 2), a * x < sin x ∧ sin x < b * x) →
  1 - 2 / π ≤ b - a :=
by sorry

end min_difference_of_sine_bounds_l3386_338611


namespace daily_production_is_1100_l3386_338663

/-- A factory produces toys with the following characteristics:
  * Produces a total of 5500 toys per week
  * Works 5 days per week
  * Produces the same number of toys each working day
-/
def ToyFactory : Type :=
  { weekly_production : ℕ // weekly_production = 5500 }
  × { working_days : ℕ // working_days = 5 }

/-- Calculate the daily toy production for a given toy factory -/
def daily_production (factory : ToyFactory) : ℕ :=
  factory.1 / factory.2

/-- Theorem stating that the daily production of toys is 1100 -/
theorem daily_production_is_1100 (factory : ToyFactory) :
  daily_production factory = 1100 := by
  sorry


end daily_production_is_1100_l3386_338663


namespace problem_statement_l3386_338612

theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 3) :
  a^4 + 1/a^4 = Real.rpow 9 (1/3) - 4 * Real.rpow 3 (1/3) + 2 := by
  sorry

end problem_statement_l3386_338612


namespace worker_c_completion_time_l3386_338691

/-- Given workers a, b, and c who can complete a work in the specified times,
    prove that c can complete the work alone in 40 days. -/
theorem worker_c_completion_time
  (total_work : ℝ)
  (time_a : ℝ) (time_b : ℝ) (time_c : ℝ)
  (total_time : ℝ) (c_left_early : ℝ)
  (h_time_a : time_a = 30)
  (h_time_b : time_b = 30)
  (h_total_time : total_time = 12)
  (h_c_left_early : c_left_early = 4)
  (h_work_completed : (total_work / time_a + total_work / time_b + total_work / time_c) *
    (total_time - c_left_early) +
    (total_work / time_a + total_work / time_b) * c_left_early = total_work) :
  time_c = 40 := by
sorry

end worker_c_completion_time_l3386_338691


namespace infinite_solutions_imply_d_equals_five_l3386_338623

theorem infinite_solutions_imply_d_equals_five :
  (∀ (d : ℝ), (∃ (S : Set ℝ), Set.Infinite S ∧ ∀ y ∈ S, 3 * (5 + d * y) = 15 * y + 15) → d = 5) :=
by sorry

end infinite_solutions_imply_d_equals_five_l3386_338623


namespace percentage_speaking_both_truth_and_lies_l3386_338670

/-- In a class with students who speak truth, lies, or both, prove the percentage
    of students speaking both truth and lies. -/
theorem percentage_speaking_both_truth_and_lies 
  (probTruth : ℝ) 
  (probLies : ℝ) 
  (probTruthOrLies : ℝ) 
  (h1 : probTruth = 0.3) 
  (h2 : probLies = 0.2) 
  (h3 : probTruthOrLies = 0.4) : 
  probTruth + probLies - probTruthOrLies = 0.1 := by
  sorry

end percentage_speaking_both_truth_and_lies_l3386_338670


namespace probability_three_common_books_l3386_338649

def total_books : ℕ := 12
def books_selected : ℕ := 6
def books_in_common : ℕ := 3

def probability_common_books : ℚ :=
  (Nat.choose total_books books_in_common * 
   Nat.choose (total_books - books_in_common) (books_selected - books_in_common) * 
   Nat.choose (total_books - books_selected) (books_selected - books_in_common)) /
  (Nat.choose total_books books_selected * Nat.choose total_books books_selected)

theorem probability_three_common_books :
  probability_common_books = 140 / 323 := by sorry

end probability_three_common_books_l3386_338649


namespace max_regions_1002_1000_l3386_338682

/-- The maximum number of regions formed by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of new regions added by each line through A after the first -/
def new_regions_per_line_A (lines_through_B : ℕ) : ℕ := lines_through_B + 2

/-- The maximum number of regions formed by m lines through A and n lines through B -/
def max_regions_two_points (m n : ℕ) : ℕ :=
  max_regions n + (new_regions_per_line_A n) + (m - 1) * (new_regions_per_line_A n)

theorem max_regions_1002_1000 :
  max_regions_two_points 1002 1000 = 1504503 := by
  sorry

end max_regions_1002_1000_l3386_338682


namespace bicycle_shop_period_l3386_338637

/-- Proves that the number of weeks that passed is 4, given the conditions of the bicycle shop problem. -/
theorem bicycle_shop_period (initial_stock : ℕ) (weekly_addition : ℕ) (sold : ℕ) (final_stock : ℕ)
  (h1 : initial_stock = 51)
  (h2 : weekly_addition = 3)
  (h3 : sold = 18)
  (h4 : final_stock = 45) :
  ∃ weeks : ℕ, weeks = 4 ∧ initial_stock + weekly_addition * weeks - sold = final_stock :=
by
  sorry

#check bicycle_shop_period

end bicycle_shop_period_l3386_338637


namespace percentage_calculation_l3386_338650

theorem percentage_calculation (number : ℝ) (h : number = 4400) : 
  0.15 * (0.30 * (0.50 * number)) = 99 := by sorry

end percentage_calculation_l3386_338650


namespace sum_of_digits_square_n_l3386_338625

/-- The number formed by repeating the digit 7 eight times -/
def n : ℕ := 77777777

/-- Sum of digits function -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem sum_of_digits_square_n : sum_of_digits (n^2) = 13 := by sorry

end sum_of_digits_square_n_l3386_338625


namespace shortest_dragon_length_l3386_338644

/-- A function that calculates the sum of digits of a positive integer -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a set of k consecutive positive integers contains a number whose digit sum is divisible by 11 -/
def isDragon (start : ℕ) (k : ℕ) : Prop :=
  ∃ i : ℕ, i < k ∧ (digitSum (start + i) % 11 = 0)

/-- The theorem stating that 39 is the smallest dragon length -/
theorem shortest_dragon_length : 
  (∀ start : ℕ, isDragon start 39) ∧ 
  (∀ k : ℕ, k < 39 → ∃ start : ℕ, ¬isDragon start k) :=
sorry

end shortest_dragon_length_l3386_338644


namespace max_perimeter_of_special_triangle_l3386_338693

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * Real.sin t.A - t.c * Real.sin t.C = (t.a - t.b) * Real.sin t.B

/-- The perimeter of the triangle -/
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- The theorem to be proved -/
theorem max_perimeter_of_special_triangle :
  ∀ t : Triangle,
    satisfiesCondition t →
    t.c = Real.sqrt 3 →
    ∃ maxPerimeter : ℝ,
      maxPerimeter = 3 * Real.sqrt 3 ∧
      ∀ t' : Triangle,
        satisfiesCondition t' →
        t'.c = Real.sqrt 3 →
        perimeter t' ≤ maxPerimeter :=
by sorry

end max_perimeter_of_special_triangle_l3386_338693
