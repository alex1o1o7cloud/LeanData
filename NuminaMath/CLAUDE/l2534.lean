import Mathlib

namespace ellipse_equation_l2534_253426

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt 3 / 2
  let perimeter := 16
  let eccentricity := (Real.sqrt (a^2 - b^2)) / a
  eccentricity = e ∧ perimeter = 4 * a → a^2 = 16 ∧ b^2 = 4 := by
  sorry

end ellipse_equation_l2534_253426


namespace derivative_of_f_l2534_253440

/-- The function f(x) = 3x^2 -/
def f (x : ℝ) : ℝ := 3 * x^2

/-- The derivative of f(x) = 3x^2 is 6x -/
theorem derivative_of_f :
  deriv f = fun x ↦ 6 * x := by sorry

end derivative_of_f_l2534_253440


namespace tylers_puppies_l2534_253436

theorem tylers_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) (total_puppies : ℕ) : 
  num_dogs = 15 → puppies_per_dog = 5 → total_puppies = num_dogs * puppies_per_dog → total_puppies = 75 :=
by
  sorry

end tylers_puppies_l2534_253436


namespace max_value_of_linear_combination_l2534_253485

theorem max_value_of_linear_combination (x y : ℝ) :
  x^2 + y^2 = 18*x + 8*y + 10 →
  4*x + 3*y ≤ 74 :=
by sorry

end max_value_of_linear_combination_l2534_253485


namespace solve_equation_l2534_253458

theorem solve_equation (X : ℝ) : (X^3)^(1/2) = 9 * 81^(1/9) → X = 3^(44/27) := by
  sorry

end solve_equation_l2534_253458


namespace teddy_has_seven_dogs_l2534_253498

/-- Represents the number of dogs Teddy has -/
def teddy_dogs : ℕ := sorry

/-- Represents the number of cats Teddy has -/
def teddy_cats : ℕ := 8

/-- Represents the number of dogs Ben has -/
def ben_dogs : ℕ := teddy_dogs + 9

/-- Represents the number of cats Dave has -/
def dave_cats : ℕ := teddy_cats + 13

/-- Represents the number of dogs Dave has -/
def dave_dogs : ℕ := teddy_dogs - 5

/-- The total number of pets -/
def total_pets : ℕ := 54

theorem teddy_has_seven_dogs : 
  teddy_dogs = 7 ∧ 
  teddy_dogs + teddy_cats + ben_dogs + dave_cats + dave_dogs = total_pets :=
sorry

end teddy_has_seven_dogs_l2534_253498


namespace share_distribution_l2534_253476

theorem share_distribution (total : ℚ) (a b c : ℚ) 
  (h1 : total = 578)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) : 
  c = 408 := by
  sorry

end share_distribution_l2534_253476


namespace power_zero_eq_one_l2534_253457

theorem power_zero_eq_one (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end power_zero_eq_one_l2534_253457


namespace room_tiling_l2534_253403

-- Define the room dimensions in centimeters
def room_length : ℕ := 544
def room_width : ℕ := 374

-- Define the function to calculate the least number of square tiles
def least_number_of_tiles (length width : ℕ) : ℕ :=
  let tile_size := Nat.gcd length width
  (length / tile_size) * (width / tile_size)

-- Theorem statement
theorem room_tiling :
  least_number_of_tiles room_length room_width = 176 := by
  sorry

end room_tiling_l2534_253403


namespace cake_and_muffin_buyers_l2534_253496

theorem cake_and_muffin_buyers (total : ℕ) (cake : ℕ) (muffin : ℕ) (neither_prob : ℚ) 
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_neither_prob : neither_prob = 1/4) :
  ∃ both : ℕ, 
    both = cake + muffin - (total * (1 - neither_prob)) ∧
    both = 15 := by
  sorry

end cake_and_muffin_buyers_l2534_253496


namespace alpha_beta_difference_bounds_l2534_253420

theorem alpha_beta_difference_bounds (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) :
  -2 < α - β ∧ α - β < 0 := by
  sorry

end alpha_beta_difference_bounds_l2534_253420


namespace simplify_expression_l2534_253429

/-- Given an expression 3(3x^2 + 4xy) - a(2x^2 + 3xy - 1), if the simplified result
    does not contain y, then a = 4 and the simplified expression is x^2 + 4 -/
theorem simplify_expression (x y : ℝ) (a : ℝ) :
  (∀ y, 3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1) = 
   3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1)) →
  a = 4 ∧ 3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1) = x^2 + 4 :=
by sorry

end simplify_expression_l2534_253429


namespace solution_to_linear_equation_with_opposite_numbers_l2534_253451

theorem solution_to_linear_equation_with_opposite_numbers :
  ∃ (x y : ℝ), 2 * x + 3 * y - 4 = 0 ∧ x = -y ∧ x = -4 ∧ y = 4 := by
  sorry

end solution_to_linear_equation_with_opposite_numbers_l2534_253451


namespace triangle_side_length_l2534_253492

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) : 
  A = π / 3 →
  Real.tan B = 1 / 2 →
  AB = 2 * Real.sqrt 3 + 1 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  AB > 0 ∧ BC > 0 ∧ AC > 0 →
  AC / Real.sin B = AB / Real.sin C →
  BC / Real.sin A = AB / Real.sin C →
  BC = Real.sqrt 15 := by
sorry


end triangle_side_length_l2534_253492


namespace hyperbola_properties_l2534_253465

/-- An equilateral hyperbola with foci on the x-axis passing through (4, -2) -/
def equilateralHyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 12

theorem hyperbola_properties :
  -- The hyperbola is equilateral
  ∀ (x y : ℝ), equilateralHyperbola x y → x^2 - y^2 = 12 ∧
  -- The foci are on the x-axis (implied by the equation form)
  -- The hyperbola passes through the point (4, -2)
  equilateralHyperbola 4 (-2) := by
  sorry

end hyperbola_properties_l2534_253465


namespace cost_is_five_l2534_253411

/-- The number of tickets available -/
def total_tickets : ℕ := 10

/-- The number of rides possible -/
def number_of_rides : ℕ := 2

/-- The cost of each ride in tickets -/
def cost_per_ride : ℕ := total_tickets / number_of_rides

/-- Theorem: The cost per ride is 5 tickets -/
theorem cost_is_five : cost_per_ride = 5 := by
  sorry

end cost_is_five_l2534_253411


namespace range_of_2a_plus_3b_inequality_with_squared_sum_l2534_253475

-- Part 1
theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) : 
  -3 ≤ 2*a + 3*b ∧ 2*a + 3*b ≤ 3 :=
sorry

-- Part 2
theorem inequality_with_squared_sum (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 = 6) : 
  1 / (a^2 + 1) + 1 / (b^2 + 2) > 1/2 - 1 / (c^2 + 3) :=
sorry

end range_of_2a_plus_3b_inequality_with_squared_sum_l2534_253475


namespace intersection_point_correct_l2534_253480

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line -/
def line1 : Line2D :=
  { point := (1, 2),
    direction := (2, -3) }

/-- The second line -/
def line2 : Line2D :=
  { point := (4, 5),
    direction := (1, -1) }

/-- A point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧
            p.2 = l.point.2 + t * l.direction.2

/-- The intersection point of the two lines -/
def intersectionPoint : ℝ × ℝ := (-11, 20)

/-- Theorem: The intersection point lies on both lines and is unique -/
theorem intersection_point_correct :
  pointOnLine intersectionPoint line1 ∧
  pointOnLine intersectionPoint line2 ∧
  ∀ p : ℝ × ℝ, pointOnLine p line1 ∧ pointOnLine p line2 → p = intersectionPoint :=
sorry

end intersection_point_correct_l2534_253480


namespace seven_students_distribution_l2534_253421

/-- The number of ways to distribute n students into two dormitories with at least m students in each -/
def distribution_plans (n : ℕ) (m : ℕ) : ℕ :=
  (Finset.sum (Finset.range (n - 2*m + 1)) (λ k => (Nat.choose n (m + k)) * 2)) * 2

/-- Theorem: There are 112 ways to distribute 7 students into two dormitories with at least 2 students in each -/
theorem seven_students_distribution : distribution_plans 7 2 = 112 := by
  sorry

end seven_students_distribution_l2534_253421


namespace fewer_sevens_100_l2534_253437

/-- A function that represents an arithmetic expression using sevens -/
def SevenExpression : Type := ℕ → ℚ

/-- Count the number of sevens used in an expression -/
def count_sevens : SevenExpression → ℕ := sorry

/-- Evaluate a SevenExpression -/
def evaluate : SevenExpression → ℚ := sorry

/-- Theorem: There exists an expression using fewer than 10 sevens that evaluates to 100 -/
theorem fewer_sevens_100 : ∃ e : SevenExpression, count_sevens e < 10 ∧ evaluate e = 100 := by sorry

end fewer_sevens_100_l2534_253437


namespace distance_to_asymptote_l2534_253462

-- Define the parabola C₁
def C₁ (a : ℝ) (x y : ℝ) : Prop := y^2 = 8*a*x ∧ a > 0

-- Define the line l
def l (a : ℝ) (x y : ℝ) : Prop := y = x - 2*a

-- Define the hyperbola C₂
def C₂ (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the directrix of C₁
def directrix (a : ℝ) : ℝ := -2*a

-- Define the focus of C₁
def focus_C₁ (a : ℝ) : ℝ × ℝ := (2*a, 0)

-- Define the asymptote of C₂
def asymptote_C₂ (a b : ℝ) (x y : ℝ) : Prop := b*x - a*y = 0

-- Main theorem
theorem distance_to_asymptote 
  (a b : ℝ) 
  (h₁ : C₁ a (2*a) 0)  -- C₁ passes through its focus
  (h₂ : ∃ x y, C₁ a x y ∧ l a x y ∧ (x - 2*a)^2 + y^2 = 256)  -- Segment length is 16
  (h₃ : ∃ x, C₂ a b x (directrix a))  -- One focus of C₂ on directrix of C₁
  : (abs (2*a)) / Real.sqrt (b^2 + a^2) = 1 :=
sorry

end distance_to_asymptote_l2534_253462


namespace path_width_l2534_253459

theorem path_width (R r : ℝ) (h1 : R > r) (h2 : 2 * π * R - 2 * π * r = 15 * π) : R - r = 7.5 := by
  sorry

end path_width_l2534_253459


namespace pattys_cafe_theorem_l2534_253448

/-- Represents the cost calculation at Patty's Cafe -/
def pattys_cafe_cost (sandwich_price soda_price discount_threshold discount : ℕ) 
                     (num_sandwiches num_sodas : ℕ) : ℕ :=
  let total_items := num_sandwiches + num_sodas
  let subtotal := sandwich_price * num_sandwiches + soda_price * num_sodas
  if total_items > discount_threshold then subtotal - discount else subtotal

/-- The cost of purchasing 7 sandwiches and 6 sodas at Patty's Cafe is $36 -/
theorem pattys_cafe_theorem : 
  pattys_cafe_cost 4 3 10 10 7 6 = 36 := by
  sorry


end pattys_cafe_theorem_l2534_253448


namespace no_real_roots_for_polynomial_l2534_253488

theorem no_real_roots_for_polynomial (a : ℝ) : 
  ¬∃ x : ℝ, x^4 + a^2*x^3 - 2*x^2 + a*x + 4 = 0 :=
by sorry

end no_real_roots_for_polynomial_l2534_253488


namespace three_digit_number_problem_l2534_253415

theorem three_digit_number_problem (a b c d e f : ℕ) :
  (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) ∧
  (100 ≤ 100 * d + 10 * e + f) ∧ (100 * d + 10 * e + f < 1000) ∧
  (a = b + 1) ∧ (b = c + 2) ∧
  ((100 * a + 10 * b + c) * 3 + 4 = 100 * d + 10 * e + f) →
  100 * d + 10 * e + f = 964 := by
  sorry


end three_digit_number_problem_l2534_253415


namespace ellipse_eccentricity_l2534_253460

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and focal distance c,
    prove that if the point symmetric to the focus with respect to y = (b/c)x lies on the ellipse,
    then the eccentricity is √2/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  let e := c / a
  let ellipse := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let focus := (c, 0)
  let symmetry_line := fun (x : ℝ) ↦ (b / c) * x
  let Q := (
    let m := (c^3 - c*b^2) / a^2
    let n := 2*b*c^2 / a^2
    (m, n)
  )
  (ellipse Q.1 Q.2) → e = Real.sqrt 2 / 2 := by
sorry

end ellipse_eccentricity_l2534_253460


namespace machine_work_time_l2534_253441

/-- The number of shirts made today -/
def shirts_today : ℕ := 8

/-- The number of shirts that can be made per minute -/
def shirts_per_minute : ℕ := 2

/-- The number of minutes the machine worked today -/
def minutes_worked : ℚ := shirts_today / shirts_per_minute

theorem machine_work_time : minutes_worked = 4 := by
  sorry

end machine_work_time_l2534_253441


namespace third_term_is_27_l2534_253477

/-- A geometric sequence with six terms where the fifth term is 81 and the sixth term is 243 -/
def geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ 
  b = a * r ∧
  c = b * r ∧
  d = c * r ∧
  81 = d * r ∧
  243 = 81 * r

/-- The third term of the geometric sequence a, b, c, d, 81, 243 is 27 -/
theorem third_term_is_27 (a b c d : ℝ) (h : geometric_sequence a b c d) : c = 27 := by
  sorry

end third_term_is_27_l2534_253477


namespace triangle_area_perimeter_inequality_triangle_area_perimeter_equality_l2534_253416

/-- Represents a triangle with area and perimeter -/
structure Triangle where
  area : ℝ
  perimeter : ℝ

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (t : Triangle) : Prop :=
  sorry -- Definition of equilateral triangle

theorem triangle_area_perimeter_inequality (t : Triangle) :
  36 * t.area ≤ t.perimeter^2 * Real.sqrt 3 :=
sorry

theorem triangle_area_perimeter_equality (t : Triangle) :
  36 * t.area = t.perimeter^2 * Real.sqrt 3 ↔ IsEquilateral t :=
sorry

end triangle_area_perimeter_inequality_triangle_area_perimeter_equality_l2534_253416


namespace kara_water_consumption_l2534_253464

/-- The amount of water Kara drinks with each medication dose in ounces -/
def water_per_dose : ℕ := 4

/-- The number of times Kara takes her medication per day -/
def doses_per_day : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of times Kara forgot to take her medication in the second week -/
def forgotten_doses : ℕ := 2

/-- The total amount of water Kara drank with her medication over two weeks -/
def total_water : ℕ := 
  (water_per_dose * doses_per_day * days_per_week) + 
  (water_per_dose * (doses_per_day * days_per_week - forgotten_doses))

theorem kara_water_consumption : total_water = 160 := by
  sorry

end kara_water_consumption_l2534_253464


namespace watercolor_pictures_after_work_l2534_253479

/-- Represents the number of papers Charles bought and used --/
structure PaperCounts where
  total_papers : ℕ
  regular_papers : ℕ
  watercolor_papers : ℕ
  today_regular : ℕ
  today_watercolor : ℕ
  yesterday_before_work : ℕ

/-- Theorem stating the number of watercolor pictures Charles drew after work yesterday --/
theorem watercolor_pictures_after_work (p : PaperCounts)
  (h1 : p.total_papers = 20)
  (h2 : p.regular_papers = 10)
  (h3 : p.watercolor_papers = 10)
  (h4 : p.today_regular = 4)
  (h5 : p.today_watercolor = 2)
  (h6 : p.yesterday_before_work = 6)
  (h7 : p.yesterday_before_work ≤ p.regular_papers)
  (h8 : p.today_regular + p.today_watercolor = 6)
  (h9 : p.regular_papers + p.watercolor_papers = p.total_papers)
  (h10 : ∃ (x : ℕ), x > 0 ∧ x ≤ p.watercolor_papers - p.today_watercolor) :
  p.watercolor_papers - p.today_watercolor = 8 := by
  sorry


end watercolor_pictures_after_work_l2534_253479


namespace arithmetic_sequence_sum_l2534_253417

/-- Given an arithmetic sequence {aₙ} with common difference d = 2 and a₄ = 3,
    prove that a₂ + a₈ = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  a 4 = 3 →                    -- a₄ = 3
  a 2 + a 8 = 10 :=             -- prove a₂ + a₈ = 10
by sorry

end arithmetic_sequence_sum_l2534_253417


namespace widget_production_difference_l2534_253404

/-- Given David's widget production scenario, this theorem proves the difference
    in production between two consecutive days. -/
theorem widget_production_difference
  (t : ℕ) -- Number of hours worked on the first day
  (w : ℕ) -- Number of widgets produced per hour on the first day
  (h1 : w = 2 * t^2) -- Relation between w and t
  : w * t - (w + 3) * (t - 3) = 6 * t^2 - 3 * t + 9 :=
by sorry

end widget_production_difference_l2534_253404


namespace fraction_scaling_l2534_253414

theorem fraction_scaling (x y : ℝ) : 
  (5*x - 5*(5*y)) / ((5*x)^2 + (5*y)^2) = (1/5) * ((x - 5*y) / (x^2 + y^2)) := by
  sorry

end fraction_scaling_l2534_253414


namespace contractor_problem_l2534_253409

-- Define the parameters
def total_days : ℕ := 50
def initial_workers : ℕ := 70
def days_passed : ℕ := 25
def work_completed : ℚ := 40 / 100

-- Define the function to calculate additional workers needed
def additional_workers_needed (total_days : ℕ) (initial_workers : ℕ) (days_passed : ℕ) (work_completed : ℚ) : ℕ :=
  -- The actual calculation will be implemented in the proof
  sorry

-- Theorem statement
theorem contractor_problem :
  additional_workers_needed total_days initial_workers days_passed work_completed = 35 :=
by
  sorry

end contractor_problem_l2534_253409


namespace sqrt_12_plus_inverse_third_plus_neg_2_squared_simplify_fraction_division_l2534_253430

-- Problem 1
theorem sqrt_12_plus_inverse_third_plus_neg_2_squared :
  Real.sqrt 12 + (-1/3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1 := by sorry

-- Problem 2
theorem simplify_fraction_division (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2*a / (a^2 - 4)) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2) := by sorry

end sqrt_12_plus_inverse_third_plus_neg_2_squared_simplify_fraction_division_l2534_253430


namespace work_completion_time_l2534_253433

theorem work_completion_time (A : ℝ) (h1 : A > 0) : 
  (∃ B : ℝ, B = A / 2 ∧ 1 / A + 1 / B = 1 / 6) → A = 18 := by
  sorry

end work_completion_time_l2534_253433


namespace f_maximum_and_a_range_l2534_253494

/-- The function f(x) = |x+1| - |x-4| - a -/
def f (x a : ℝ) : ℝ := |x + 1| - |x - 4| - a

theorem f_maximum_and_a_range :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x a ≤ f x_max a) ∧
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x a ≤ 5 - a) ∧
  (∃ (x : ℝ), f x a ≥ 4/a + 1 → (a = 2 ∨ a < 0)) := by
  sorry

end f_maximum_and_a_range_l2534_253494


namespace limit_f_at_zero_l2534_253445

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.exp x - 5 * x) / (4 * x^2 + 7 * x)

theorem limit_f_at_zero : 
  Filter.Tendsto f (Filter.atTop.comap (fun x => 1 / x)) (nhds (-4/7)) := by
  sorry

end limit_f_at_zero_l2534_253445


namespace snooker_tournament_revenue_l2534_253452

theorem snooker_tournament_revenue :
  let total_tickets : ℕ := 320
  let vip_price : ℚ := 40
  let general_price : ℚ := 10
  let vip_tickets : ℕ := (total_tickets - 148) / 2
  let general_tickets : ℕ := (total_tickets + 148) / 2
  (vip_price * vip_tickets + general_price * general_tickets : ℚ) = 5780 :=
by sorry

end snooker_tournament_revenue_l2534_253452


namespace sandbag_weight_l2534_253422

/-- Calculates the weight of a partially filled sandbag with a heavier material -/
theorem sandbag_weight (capacity : ℝ) (fill_percentage : ℝ) (weight_increase : ℝ) : 
  capacity = 450 →
  fill_percentage = 0.75 →
  weight_increase = 0.65 →
  capacity * fill_percentage * (1 + weight_increase) = 556.875 := by
  sorry

end sandbag_weight_l2534_253422


namespace intersected_cubes_count_l2534_253482

/-- Represents a 3D coordinate --/
structure Coord :=
  (x y z : ℕ)

/-- Represents a cube --/
structure Cube :=
  (side_length : ℕ)

/-- Represents a plane perpendicular to the main diagonal of a cube --/
structure DiagonalPlane :=
  (cube : Cube)
  (passes_through_center : Bool)

/-- Counts the number of unit cubes intersected by a diagonal plane in a larger cube --/
def count_intersected_cubes (c : Cube) (p : DiagonalPlane) : ℕ :=
  sorry

/-- The main theorem to be proved --/
theorem intersected_cubes_count (c : Cube) (p : DiagonalPlane) :
  c.side_length = 5 →
  p.cube = c →
  p.passes_through_center = true →
  count_intersected_cubes c p = 55 :=
sorry

end intersected_cubes_count_l2534_253482


namespace equation_solution_l2534_253443

theorem equation_solution : 
  let f (x : ℝ) := 1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7))
  ∀ x : ℝ, f x = 1/8 ↔ x = -2 * Real.sqrt 14 ∨ x = 2 * Real.sqrt 14 := by
  sorry

end equation_solution_l2534_253443


namespace fraction_equality_l2534_253497

theorem fraction_equality (x y : ℝ) (h : x ≠ -y) : (-x + y) / (-x - y) = (x - y) / (x + y) := by
  sorry

end fraction_equality_l2534_253497


namespace prob_through_c_is_three_sevenths_l2534_253418

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths between two points -/
def numPaths (start finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of passing through a point when moving from start to finish -/
def probThroughPoint (start mid finish : Point) : Rat :=
  (numPaths start mid * numPaths mid finish : Rat) / numPaths start finish

theorem prob_through_c_is_three_sevenths : 
  let a := Point.mk 0 0
  let b := Point.mk 4 4
  let c := Point.mk 3 2
  probThroughPoint a c b = 3/7 := by
  sorry

end prob_through_c_is_three_sevenths_l2534_253418


namespace value_of_z_l2534_253453

theorem value_of_z (z : ℝ) : 
  (Real.sqrt 1.1) / (Real.sqrt 0.81) + (Real.sqrt z) / (Real.sqrt 0.49) = 2.879628878919216 → 
  z = 1.44 := by
sorry

end value_of_z_l2534_253453


namespace egg_distribution_proof_l2534_253432

def mia_eggs : ℕ := 4
def sofia_eggs : ℕ := 2 * mia_eggs
def pablo_eggs : ℕ := 4 * sofia_eggs

def total_eggs : ℕ := mia_eggs + sofia_eggs + pablo_eggs
def equal_distribution : ℚ := total_eggs / 3

def fraction_to_sofia : ℚ := 5 / 24

theorem egg_distribution_proof :
  let sofia_new := sofia_eggs + (fraction_to_sofia * pablo_eggs)
  let mia_new := equal_distribution
  let pablo_new := pablo_eggs - (fraction_to_sofia * pablo_eggs) - (mia_new - mia_eggs)
  sofia_new = mia_new ∧ sofia_new = pablo_new := by sorry

end egg_distribution_proof_l2534_253432


namespace rectangle_properties_l2534_253450

structure Quadrilateral where
  isRectangle : Bool
  diagonalsEqual : Bool
  diagonalsBisect : Bool

theorem rectangle_properties (q : Quadrilateral) :
  (q.isRectangle → q.diagonalsEqual ∧ q.diagonalsBisect) ∧
  (q.diagonalsEqual ∧ q.diagonalsBisect → q.isRectangle) ∧
  (¬q.isRectangle → ¬q.diagonalsEqual ∨ ¬q.diagonalsBisect) ∧
  (¬q.diagonalsEqual ∨ ¬q.diagonalsBisect → ¬q.isRectangle) :=
by sorry

end rectangle_properties_l2534_253450


namespace smallest_number_of_eggs_l2534_253424

/-- The number of eggs in a full container -/
def full_container : ℕ := 12

/-- The number of containers with missing eggs -/
def containers_with_missing : ℕ := 2

/-- The number of eggs missing from each incomplete container -/
def eggs_missing_per_container : ℕ := 1

/-- The minimum number of eggs we're looking for -/
def min_eggs : ℕ := 106

theorem smallest_number_of_eggs :
  ∀ n : ℕ,
  n > 100 ∧
  ∃ c : ℕ, n = c * full_container - containers_with_missing * eggs_missing_per_container →
  n ≥ min_eggs :=
by sorry

end smallest_number_of_eggs_l2534_253424


namespace probability_of_even_product_l2534_253491

-- Define the set of chips in each box
def chips : Set ℕ := {1, 2, 4}

-- Define the function to check if a number is even
def isEven (n : ℕ) : Prop := n % 2 = 0

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := 27

-- Define the number of favorable outcomes (even products)
def favorableOutcomes : ℕ := 26

-- Theorem statement
theorem probability_of_even_product :
  (favorableOutcomes : ℚ) / totalOutcomes = 26 / 27 := by sorry

end probability_of_even_product_l2534_253491


namespace inequalities_comparison_l2534_253401

theorem inequalities_comparison (a b : ℝ) : 
  (∃ a b : ℝ, a + b < 2) ∧ 
  (∀ a b : ℝ, a^2 + b^2 ≥ 2*a*b) ∧ 
  (∀ a b : ℝ, a*b ≤ ((a + b)/2)^2) ∧ 
  (∀ a b : ℝ, |a| + |b| ≥ 2) :=
by sorry

end inequalities_comparison_l2534_253401


namespace xy_values_l2534_253478

theorem xy_values (x y : ℝ) 
  (h1 : (16:ℝ)^x / (4:ℝ)^(x+y) = 64)
  (h2 : (4:ℝ)^(x+y) / (2:ℝ)^(5*y) = 16) :
  x = 5 ∧ y = 2 := by
  sorry

end xy_values_l2534_253478


namespace soccer_tournament_equation_l2534_253470

/-- Represents a soccer invitational tournament -/
structure SoccerTournament where
  num_teams : ℕ
  num_matches : ℕ
  each_pair_plays : Bool

/-- The equation for the number of matches in the tournament -/
def tournament_equation (t : SoccerTournament) : Prop :=
  t.num_matches = (t.num_teams * (t.num_teams - 1)) / 2

/-- Theorem stating the correct equation for the given tournament conditions -/
theorem soccer_tournament_equation (t : SoccerTournament) 
  (h1 : t.each_pair_plays = true) 
  (h2 : t.num_matches = 28) : 
  tournament_equation t :=
sorry

end soccer_tournament_equation_l2534_253470


namespace correct_statement_l2534_253467

-- Define proposition P
def P : Prop := ∀ x : ℝ, x^2 - 4*x + 5 > 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, x > 0 ∧ Real.cos x > 1

-- Theorem to prove
theorem correct_statement : P ∨ ¬q := by
  sorry

end correct_statement_l2534_253467


namespace existence_of_special_divisor_l2534_253493

theorem existence_of_special_divisor (n k : ℕ) (h1 : n > 1) (h2 : k = (Nat.factors n).card) :
  ∃ a : ℕ, 1 < a ∧ a < n / k + 1 ∧ n ∣ (a^2 - a) :=
sorry

end existence_of_special_divisor_l2534_253493


namespace min_reciprocal_sum_l2534_253474

theorem min_reciprocal_sum (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1/x + 1/y + 1/z ≥ 3 ∧ ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 3 ∧ 1/a + 1/b + 1/c = 3 :=
sorry

end min_reciprocal_sum_l2534_253474


namespace equivalence_theorem_l2534_253468

-- Define f as a differentiable function on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the condition that f(x) + f'(x) > 0 for all x ∈ ℝ
variable (h : ∀ x : ℝ, f x + (deriv f) x > 0)

-- State the theorem
theorem equivalence_theorem (a b : ℝ) :
  a > b ↔ Real.exp a * f a > Real.exp b * f b :=
sorry

end equivalence_theorem_l2534_253468


namespace geometric_sum_five_terms_l2534_253499

theorem geometric_sum_five_terms :
  (1 / 5 : ℚ) - (1 / 25 : ℚ) + (1 / 125 : ℚ) - (1 / 625 : ℚ) + (1 / 3125 : ℚ) = 521 / 3125 := by
  sorry

end geometric_sum_five_terms_l2534_253499


namespace power_of_power_l2534_253487

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end power_of_power_l2534_253487


namespace rectangular_hall_area_l2534_253490

theorem rectangular_hall_area (length width : ℝ) : 
  width = length / 2 →
  length - width = 8 →
  length * width = 128 :=
by
  sorry

end rectangular_hall_area_l2534_253490


namespace golden_comets_ratio_l2534_253461

/-- Represents the number of chickens in a flock -/
structure ChickenFlock where
  rhodeIslandReds : ℕ
  goldenComets : ℕ

/-- Given information about Susie's and Britney's chicken flocks -/
def susie : ChickenFlock := { rhodeIslandReds := 11, goldenComets := 6 }
def britney : ChickenFlock :=
  { rhodeIslandReds := 2 * susie.rhodeIslandReds,
    goldenComets := susie.rhodeIslandReds + susie.goldenComets + 8 - (2 * susie.rhodeIslandReds) }

/-- The theorem to be proved -/
theorem golden_comets_ratio :
  2 * britney.goldenComets = susie.goldenComets := by sorry

end golden_comets_ratio_l2534_253461


namespace digit_2009_is_zero_l2534_253402

/-- The sequence of digits obtained by writing natural numbers successively -/
def digit_sequence : ℕ → ℕ := sorry

/-- The number of digits used to write numbers from 1 to n -/
def digits_count (n : ℕ) : ℕ := sorry

/-- The 2009th digit in the sequence -/
def digit_2009 : ℕ := digit_sequence 2009

theorem digit_2009_is_zero : digit_2009 = 0 := by sorry

end digit_2009_is_zero_l2534_253402


namespace carnation_fraction_l2534_253446

/-- Represents a flower bouquet with pink and red roses and carnations -/
structure Bouquet where
  pink_roses : ℚ
  red_roses : ℚ
  pink_carnations : ℚ
  red_carnations : ℚ

/-- The fraction of carnations in the bouquet is 7/10 -/
theorem carnation_fraction (b : Bouquet) : 
  b.pink_roses + b.red_roses + b.pink_carnations + b.red_carnations = 1 →
  b.pink_roses + b.pink_carnations = 6/10 →
  b.pink_roses = 1/3 * (b.pink_roses + b.pink_carnations) →
  b.red_carnations = 3/4 * (b.red_roses + b.red_carnations) →
  b.pink_carnations + b.red_carnations = 7/10 := by
  sorry

end carnation_fraction_l2534_253446


namespace some_number_value_l2534_253455

theorem some_number_value (x y z w N : ℝ) 
  (h1 : 4 * x * z + y * w = N)
  (h2 : x * w + y * z = 6)
  (h3 : (2 * x + y) * (2 * z + w) = 15) :
  N = 3 := by sorry

end some_number_value_l2534_253455


namespace probability_of_E_l2534_253469

def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

def count (c : Char) : ℕ :=
  match c with
  | 'A' => 5
  | 'E' => 3
  | 'I' => 4
  | 'O' => 2
  | 'U' => 6
  | _ => 0

def total_count : ℕ := (vowels.sum count)

theorem probability_of_E : 
  (count 'E' : ℚ) / total_count = 3 / 20 := by sorry

end probability_of_E_l2534_253469


namespace problem_solution_l2534_253428

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) :
  (x - 3)^2 + 16 / (x - 3)^2 = 23 := by sorry

end problem_solution_l2534_253428


namespace x_plus_y_value_l2534_253405

theorem x_plus_y_value (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 2) (h3 : x * y < 0) :
  x + y = 1 ∨ x + y = -1 := by
  sorry

end x_plus_y_value_l2534_253405


namespace corn_acreage_l2534_253406

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) : 
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end corn_acreage_l2534_253406


namespace smallest_m_exceeds_15_l2534_253435

def sum_digits_after_decimal (n : ℚ) : ℕ :=
  sorry

def exceeds_15 (m : ℕ) : Prop :=
  sum_digits_after_decimal (1 / 3^m) > 15

theorem smallest_m_exceeds_15 : 
  (∀ k < 7, ¬(exceeds_15 k)) ∧ exceeds_15 7 :=
sorry

end smallest_m_exceeds_15_l2534_253435


namespace remainder_4053_div_23_l2534_253419

theorem remainder_4053_div_23 : 4053 % 23 = 5 := by
  sorry

end remainder_4053_div_23_l2534_253419


namespace amount_calculation_l2534_253412

theorem amount_calculation (a b : ℝ) 
  (h1 : a + b = 1210)
  (h2 : (1/3) * a = (1/4) * b) : 
  b = 4840 / 7 := by
  sorry

end amount_calculation_l2534_253412


namespace custom_op_example_l2534_253442

/-- Custom binary operation @ defined as a @ b = 5a - 2b -/
def custom_op (a b : ℝ) : ℝ := 5 * a - 2 * b

/-- Theorem stating that 4 @ 7 = 6 under the custom operation -/
theorem custom_op_example : custom_op 4 7 = 6 := by
  sorry

end custom_op_example_l2534_253442


namespace half_abs_diff_squares_20_15_l2534_253449

theorem half_abs_diff_squares_20_15 : (1 / 2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end half_abs_diff_squares_20_15_l2534_253449


namespace lion_population_l2534_253472

/-- Given a population of lions that increases by 4 each month for 12 months,
    prove that if the final population is 148, the initial population was 100. -/
theorem lion_population (initial_population final_population : ℕ) 
  (monthly_increase : ℕ) (months : ℕ) : 
  monthly_increase = 4 →
  months = 12 →
  final_population = 148 →
  final_population = initial_population + monthly_increase * months →
  initial_population = 100 := by
sorry

end lion_population_l2534_253472


namespace a_86_in_geometric_subsequence_l2534_253484

/-- Represents an arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_nonzero : d ≠ 0)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d)

/-- Represents a subsequence of an arithmetic sequence that forms a geometric sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) :=
  (k : ℕ → ℕ)
  (h_geometric : ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, as.a (k (n + 1)) = r * as.a (k n))
  (h_k1 : k 1 = 1)
  (h_k2 : k 2 = 2)
  (h_k3 : k 3 = 6)

/-- The main theorem stating that a_86 is in the geometric subsequence -/
theorem a_86_in_geometric_subsequence (as : ArithmeticSequence) (gs : GeometricSubsequence as) :
  ∃ n : ℕ, gs.k n = 86 :=
sorry

end a_86_in_geometric_subsequence_l2534_253484


namespace solution_proof_l2534_253447

-- Part 1: System of equations
def satisfies_system (x y : ℝ) : Prop :=
  2 * x - y = 5 ∧ 3 * x + 4 * y = 2

-- Part 2: System of inequalities
def satisfies_inequalities (x : ℝ) : Prop :=
  -2 * x < 6 ∧ 3 * (x - 2) ≤ x - 4

-- Part 3: Integer solutions
def is_integer_solution (x : ℤ) : Prop :=
  -3 < (x : ℝ) ∧ (x : ℝ) ≤ 1

theorem solution_proof :
  -- Part 1
  satisfies_system 2 (-1) ∧
  -- Part 2
  (∀ x : ℝ, satisfies_inequalities x ↔ -3 < x ∧ x ≤ 1) ∧
  -- Part 3
  (∀ x : ℤ, is_integer_solution x ↔ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1) :=
by sorry

end solution_proof_l2534_253447


namespace line_x_intercept_l2534_253434

/-- A straight line passing through two points (2, -3) and (6, 5) has an x-intercept of 7/2 -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ),
  (f 2 = -3) →
  (f 6 = 5) →
  (∀ x y : ℝ, f y - f x = (y - x) * ((5 - (-3)) / (6 - 2))) →
  (∃ x : ℝ, f x = 0 ∧ x = 7/2) :=
by
  sorry

end line_x_intercept_l2534_253434


namespace option2_expected_cost_l2534_253423

/-- Represents the water temperature situations -/
inductive WaterTemp
  | Normal
  | SlightlyHigh
  | ExtremelyHigh

/-- Probability of extremely high water temperature -/
def probExtremelyHigh : ℝ := 0.01

/-- Probability of slightly high water temperature -/
def probSlightlyHigh : ℝ := 0.25

/-- Loss incurred when water temperature is extremely high -/
def lossExtremelyHigh : ℝ := 600000

/-- Loss incurred when water temperature is slightly high -/
def lossSlightlyHigh : ℝ := 100000

/-- Cost of implementing Option 2 (temperature control equipment) -/
def costOption2 : ℝ := 20000

/-- Expected cost of Option 2 -/
def expectedCostOption2 : ℝ := 
  (lossExtremelyHigh + costOption2) * probExtremelyHigh + costOption2 * (1 - probExtremelyHigh)

/-- Theorem stating that the expected cost of Option 2 is 2600 yuan -/
theorem option2_expected_cost : expectedCostOption2 = 2600 := by
  sorry

end option2_expected_cost_l2534_253423


namespace negation_of_forall_squared_plus_one_nonnegative_l2534_253483

theorem negation_of_forall_squared_plus_one_nonnegative :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + 1 < 0) := by sorry

end negation_of_forall_squared_plus_one_nonnegative_l2534_253483


namespace sum_of_special_numbers_l2534_253413

/-- A natural number that ends in 5 zeros and has exactly 42 divisors -/
def SpecialNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^5 * k ∧ (Nat.divisors n).card = 42

/-- The theorem stating that there are exactly two distinct natural numbers
    that satisfy the SpecialNumber property, and their sum is 700000 -/
theorem sum_of_special_numbers :
  ∃! (a b : ℕ), a < b ∧ SpecialNumber a ∧ SpecialNumber b ∧ a + b = 700000 := by
  sorry

#check sum_of_special_numbers

end sum_of_special_numbers_l2534_253413


namespace square_difference_given_linear_equations_l2534_253400

theorem square_difference_given_linear_equations (x y : ℝ) :
  (3 * x + 2 * y = 30) → (4 * x + 2 * y = 34) → x^2 - y^2 = -65 := by
  sorry

end square_difference_given_linear_equations_l2534_253400


namespace chord_length_polar_l2534_253427

/-- The length of the chord intercepted by a line on a circle in polar coordinates -/
theorem chord_length_polar (r : ℝ) (h : r > 0) :
  let line := {θ : ℝ | r * (Real.sin θ + Real.cos θ) = 2 * Real.sqrt 2}
  let circle := {ρ : ℝ | ρ = 2 * Real.sqrt 2}
  let chord_length := 2 * Real.sqrt ((2 * Real.sqrt 2)^2 - (2 * Real.sqrt 2 / Real.sqrt 2)^2)
  chord_length = 4 := by
  sorry

end chord_length_polar_l2534_253427


namespace unique_student_count_l2534_253456

/-- Represents the number of students that can be seated in large boats -/
def large_boat_capacity (num_boats : ℕ) : ℕ := 17 * num_boats + 6

/-- Represents the number of students that can be seated in small boats -/
def small_boat_capacity (num_boats : ℕ) : ℕ := 10 * num_boats + 2

/-- Theorem stating that 142 is the only number of students satisfying all conditions -/
theorem unique_student_count : 
  ∃! n : ℕ, 
    100 < n ∧ 
    n < 200 ∧ 
    (∃ x y : ℕ, large_boat_capacity x = n ∧ small_boat_capacity y = n) :=
by
  sorry

#check unique_student_count

end unique_student_count_l2534_253456


namespace max_people_in_line_l2534_253425

/-- Represents the state of the line at any given point -/
structure LineState where
  current : ℕ
  max : ℕ

/-- Updates the line state after people leave and join -/
def updateLine (state : LineState) (leave : ℕ) (join : ℕ) : LineState :=
  let remaining := state.current - min state.current leave
  let newCurrent := remaining + join
  { current := newCurrent, max := max state.max newCurrent }

/-- Repeats the process of people leaving and joining for a given number of times -/
def repeatProcess (initialState : LineState) (leave : ℕ) (join : ℕ) (times : ℕ) : LineState :=
  match times with
  | 0 => initialState
  | n + 1 => repeatProcess (updateLine initialState leave join) leave join n

/-- Calculates the final state after the entire process -/
def finalState (initial : ℕ) (leave : ℕ) (join : ℕ) (repetitions : ℕ) : LineState :=
  let initialState : LineState := { current := initial, max := initial }
  let afterRepetitions := repeatProcess initialState leave join repetitions
  let additionalJoin := afterRepetitions.current / 10  -- 10% rounded down
  updateLine afterRepetitions 0 additionalJoin

/-- Theorem stating that the maximum number of people in line is equal to the initial number -/
theorem max_people_in_line (initial : ℕ) (leave : ℕ) (join : ℕ) (repetitions : ℕ) 
    (h_initial : initial = 9) (h_leave : leave = 6) (h_join : join = 3) (h_repetitions : repetitions = 3) :
    (finalState initial leave join repetitions).max = initial := by
  sorry

end max_people_in_line_l2534_253425


namespace fuel_mixture_problem_l2534_253431

/-- Proves that 82 gallons of fuel A were added to a 208-gallon tank -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 208 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), fuel_a = 82 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end fuel_mixture_problem_l2534_253431


namespace inequality_implication_l2534_253486

theorem inequality_implication (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_ineq : 4 * Real.log x + 2 * Real.log (2 * y) ≥ x^2 + 8 * y - 4) : 
  x * y = Real.sqrt 2 / 4 ∧ x + 2 * y = 1 / 2 + Real.sqrt 2 := by
  sorry

end inequality_implication_l2534_253486


namespace x_varies_as_square_root_of_z_l2534_253471

/-- If x varies as the square of y, and y varies as the fourth root of z,
    then x varies as the square root of z. -/
theorem x_varies_as_square_root_of_z
  (k : ℝ) (j : ℝ) (x y z : ℝ → ℝ)
  (h1 : ∀ t, x t = k * (y t)^2)
  (h2 : ∀ t, y t = j * (z t)^(1/4)) :
  ∃ m : ℝ, ∀ t, x t = m * (z t)^(1/2) :=
sorry

end x_varies_as_square_root_of_z_l2534_253471


namespace sixth_edge_possibilities_l2534_253410

/-- Represents the edge lengths of a tetrahedron -/
structure TetrahedronEdges :=
  (a b c d e f : ℕ)

/-- Checks if three lengths satisfy the triangle inequality -/
def satisfiesTriangleInequality (x y z : ℕ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

/-- Checks if all faces of a tetrahedron satisfy the triangle inequality -/
def validTetrahedron (t : TetrahedronEdges) : Prop :=
  satisfiesTriangleInequality t.a t.b t.c ∧
  satisfiesTriangleInequality t.a t.d t.e ∧
  satisfiesTriangleInequality t.b t.d t.f ∧
  satisfiesTriangleInequality t.c t.e t.f

/-- The main theorem stating that there are exactly 6 possible lengths for the sixth edge -/
theorem sixth_edge_possibilities :
  ∃! (s : Finset ℕ),
    s.card = 6 ∧
    (∀ x, x ∈ s ↔ ∃ t : TetrahedronEdges,
      t.a = 14 ∧ t.b = 20 ∧ t.c = 40 ∧ t.d = 52 ∧ t.e = 70 ∧ t.f = x ∧
      validTetrahedron t) :=
by sorry


end sixth_edge_possibilities_l2534_253410


namespace cyclist_return_speed_l2534_253466

theorem cyclist_return_speed 
  (total_distance : ℝ) 
  (first_segment : ℝ) 
  (second_segment : ℝ) 
  (first_speed : ℝ) 
  (second_speed : ℝ) 
  (total_time : ℝ) :
  total_distance = first_segment + second_segment →
  first_segment = 12 →
  second_segment = 24 →
  first_speed = 8 →
  second_speed = 12 →
  total_time = 7.5 →
  (total_distance / first_speed + second_segment / second_speed + 
   (total_distance / ((total_time - (total_distance / first_speed + second_segment / second_speed))))) = total_time →
  (total_distance / (total_time - (total_distance / first_speed + second_segment / second_speed))) = 9 := by
sorry

end cyclist_return_speed_l2534_253466


namespace inner_hexagon_area_l2534_253463

/-- Represents an equilateral triangle --/
structure EquilateralTriangle where
  sideLength : ℝ
  area : ℝ

/-- Represents the configuration of triangles in the problem --/
structure TriangleConfiguration where
  largeTriangle : EquilateralTriangle
  smallTriangles : List EquilateralTriangle
  innerHexagonArea : ℝ

/-- The given configuration satisfies the problem conditions --/
def satisfiesProblemConditions (config : TriangleConfiguration) : Prop :=
  config.smallTriangles.length = 6 ∧
  config.smallTriangles.map (λ t => t.area) = [1, 1, 9, 9, 16, 16]

/-- The theorem to be proved --/
theorem inner_hexagon_area 
  (config : TriangleConfiguration) 
  (h : satisfiesProblemConditions config) : 
  config.innerHexagonArea = 38 := by
  sorry

end inner_hexagon_area_l2534_253463


namespace triangle_area_equalities_l2534_253407

theorem triangle_area_equalities (S r R A B C : ℝ) 
  (h_positive : S > 0 ∧ r > 0 ∧ R > 0)
  (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_area : S = r * R * (Real.sin A + Real.sin B + Real.sin C)) :
  S = r * R * (Real.sin A + Real.sin B + Real.sin C) ∧
  S = 4 * r * R * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2) ∧
  S = (R^2 / 2) * (Real.sin (2*A) + Real.sin (2*B) + Real.sin (2*C)) ∧
  S = 2 * R^2 * Real.sin A * Real.sin B * Real.sin C := by
  sorry

end triangle_area_equalities_l2534_253407


namespace arrangement_probability_l2534_253408

def total_tiles : ℕ := 8
def x_tiles : ℕ := 5
def o_tiles : ℕ := 3

def specific_arrangement : List Char := ['X', 'X', 'X', 'O', 'O', 'X', 'O', 'X']

def probability_of_arrangement : ℚ := 1 / 56

theorem arrangement_probability :
  probability_of_arrangement = 1 / (total_tiles.choose x_tiles) :=
sorry

end arrangement_probability_l2534_253408


namespace complex_expression_equality_l2534_253454

theorem complex_expression_equality (a b : ℂ) :
  a = 3 + 2*I ∧ b = 2 - 3*I →
  3*a + 4*b + a^2 + b^2 = 35 - 6*I :=
by sorry

end complex_expression_equality_l2534_253454


namespace function_is_constant_l2534_253495

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f is constant. -/
theorem function_is_constant (f : ℝ → ℝ) (a : ℝ) (ha : a > 0)
  (h1 : ∀ x, 0 < f x ∧ f x ≤ a)
  (h2 : ∀ x y, Real.sqrt (f x * f y) ≥ f ((x + y) / 2)) :
  ∃ c, ∀ x, f x = c :=
sorry

end function_is_constant_l2534_253495


namespace shoeing_time_for_48_blacksmiths_60_horses_l2534_253481

/-- The minimum time required for a group of blacksmiths to shoe a group of horses -/
def minimum_shoeing_time (num_blacksmiths : ℕ) (num_horses : ℕ) (time_per_horseshoe : ℕ) : ℕ :=
  let total_horseshoes := num_horses * 4
  let total_time := total_horseshoes * time_per_horseshoe
  total_time / num_blacksmiths

theorem shoeing_time_for_48_blacksmiths_60_horses : 
  minimum_shoeing_time 48 60 5 = 25 := by
  sorry

#eval minimum_shoeing_time 48 60 5

end shoeing_time_for_48_blacksmiths_60_horses_l2534_253481


namespace average_speed_calculation_average_speed_approximation_l2534_253444

theorem average_speed_calculation (total_distance : ℝ) (first_segment_distance : ℝ) 
  (first_segment_speed : ℝ) (second_segment_distance : ℝ) (second_segment_speed_limit : ℝ) 
  (second_segment_normal_speed : ℝ) (third_segment_distance : ℝ) 
  (speed_limit_distance : ℝ) : ℝ :=
  let first_segment_time := first_segment_distance / first_segment_speed
  let second_segment_time_limit := speed_limit_distance / second_segment_speed_limit
  let second_segment_time_normal := (second_segment_distance - speed_limit_distance) / second_segment_normal_speed
  let second_segment_time := second_segment_time_limit + second_segment_time_normal
  let third_segment_time := first_segment_time * 2.5
  let total_time := first_segment_time + second_segment_time + third_segment_time
  let average_speed := total_distance / total_time
  average_speed

#check average_speed_calculation 760 320 80 240 45 60 200 100

theorem average_speed_approximation :
  ∃ ε > 0, abs (average_speed_calculation 760 320 80 240 45 60 200 100 - 40.97) < ε :=
sorry

end average_speed_calculation_average_speed_approximation_l2534_253444


namespace log_inequality_equivalence_l2534_253438

-- Define the logarithm with base 1/3
noncomputable def log_one_third (x : ℝ) : ℝ := Real.log x / Real.log (1/3)

-- State the theorem
theorem log_inequality_equivalence :
  ∀ x : ℝ, log_one_third (2*x - 1) > 1 ↔ 1/2 < x ∧ x < 2/3 :=
by sorry

end log_inequality_equivalence_l2534_253438


namespace danai_decorations_l2534_253489

/-- The number of decorations Danai will put up in total -/
def total_decorations (skulls broomsticks spiderwebs pumpkins cauldron budget_left left_to_put_up : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_left + left_to_put_up

/-- Theorem stating the total number of decorations Danai will put up -/
theorem danai_decorations :
  ∀ (skulls broomsticks spiderwebs pumpkins cauldron budget_left left_to_put_up : ℕ),
    skulls = 12 →
    broomsticks = 4 →
    spiderwebs = 12 →
    pumpkins = 2 * spiderwebs →
    cauldron = 1 →
    budget_left = 20 →
    left_to_put_up = 10 →
    total_decorations skulls broomsticks spiderwebs pumpkins cauldron budget_left left_to_put_up = 83 :=
by sorry

end danai_decorations_l2534_253489


namespace sum_cannot_have_all_odd_digits_l2534_253473

/-- A digit is a natural number between 0 and 9. -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- A sequence of 1001 digits. -/
def DigitSequence : Type := Fin 1001 → Digit

/-- The first number formed by the digit sequence. -/
def firstNumber (a : DigitSequence) : ℕ := sorry

/-- The second number formed by the reversed digit sequence. -/
def secondNumber (a : DigitSequence) : ℕ := sorry

/-- A number has all odd digits if each of its digits is odd. -/
def hasAllOddDigits (n : ℕ) : Prop := sorry

theorem sum_cannot_have_all_odd_digits (a : DigitSequence) :
  ¬(hasAllOddDigits (firstNumber a + secondNumber a)) :=
sorry

end sum_cannot_have_all_odd_digits_l2534_253473


namespace hyperbola_eccentricity_eccentricity_is_four_l2534_253439

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity : ℝ → ℝ → ℝ → Prop :=
  fun a b e =>
    -- Hyperbola equation
    (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 →
    -- Parabola equation
    ∃ x₀, ∀ y, y^2 = 16 * x₀ →
    -- Right focus of hyperbola coincides with focus of parabola
    4 = (a^2 + b^2).sqrt →
    -- Eccentricity definition
    e = (a^2 + b^2).sqrt / a →
    -- Prove eccentricity is 4
    e = 4)

/-- The main theorem stating the eccentricity is 4 -/
theorem eccentricity_is_four :
  ∃ a b e, hyperbola_eccentricity a b e :=
sorry

end hyperbola_eccentricity_eccentricity_is_four_l2534_253439
