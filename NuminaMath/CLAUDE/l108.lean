import Mathlib

namespace safe_code_count_l108_10848

/-- The set of digits from 0 to 9 -/
def Digits : Finset ℕ := Finset.range 10

/-- The length of the safe code -/
def CodeLength : ℕ := 4

/-- The set of forbidden first digits -/
def ForbiddenFirstDigits : Finset ℕ := {5, 7}

/-- The number of valid safe codes -/
def ValidCodes : ℕ := 10^CodeLength - ForbiddenFirstDigits.card * 10^(CodeLength - 1)

theorem safe_code_count : ValidCodes = 9900 := by
  sorry

end safe_code_count_l108_10848


namespace tile_size_calculation_l108_10880

theorem tile_size_calculation (length width : ℝ) (num_tiles : ℕ) (h1 : length = 2) (h2 : width = 12) (h3 : num_tiles = 6) :
  (length * width) / num_tiles = 4 := by
  sorry

end tile_size_calculation_l108_10880


namespace orange_bin_problem_l108_10891

theorem orange_bin_problem (initial_oranges : ℕ) : 
  initial_oranges - 2 + 28 = 31 → initial_oranges = 5 := by
  sorry

end orange_bin_problem_l108_10891


namespace smallest_value_u_cube_plus_v_cube_l108_10866

theorem smallest_value_u_cube_plus_v_cube (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 2)
  (h2 : Complex.abs (u^2 + v^2) = 17) :
  Complex.abs (u^3 + v^3) = 47 := by
  sorry

end smallest_value_u_cube_plus_v_cube_l108_10866


namespace perimeter_of_similar_triangle_l108_10841

/-- Represents a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Determines if two triangles are similar -/
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem perimeter_of_similar_triangle (abc pqr : Triangle) :
  abc.a = abc.b ∧ 
  abc.a = 12 ∧ 
  abc.c = 14 ∧ 
  similar abc pqr ∧
  max pqr.a (max pqr.b pqr.c) = 35 →
  perimeter pqr = 95 := by
  sorry

end perimeter_of_similar_triangle_l108_10841


namespace max_area_rectangle_l108_10812

/-- Given a rectangle with integer side lengths and a perimeter of 150 feet,
    the maximum possible area is 1406 square feet. -/
theorem max_area_rectangle (x y : ℕ) : 
  x + y = 75 → x * y ≤ 1406 :=
by sorry

end max_area_rectangle_l108_10812


namespace f_at_2_l108_10831

-- Define the polynomial function
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Theorem statement
theorem f_at_2 : f 2 = 62 := by
  sorry

end f_at_2_l108_10831


namespace salary_expenditure_l108_10846

theorem salary_expenditure (salary : ℝ) (rent_fraction : ℝ) (clothes_fraction : ℝ) (remaining : ℝ) 
  (h1 : salary = 170000)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : remaining = 17000)
  (h5 : remaining / salary + rent_fraction + clothes_fraction < 1) :
  let food_fraction := 1 - (remaining / salary + rent_fraction + clothes_fraction)
  food_fraction = 1/5 := by
sorry

end salary_expenditure_l108_10846


namespace power_multiplication_l108_10862

theorem power_multiplication (t : ℝ) : t^3 * t^4 = t^7 := by
  sorry

end power_multiplication_l108_10862


namespace triangle_angle_problem_l108_10849

theorem triangle_angle_problem (A B C : Real) (a b c : Real) :
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = c) →
  (C = π / 5) →
  (B = 3 * π / 10) := by
  sorry

end triangle_angle_problem_l108_10849


namespace min_value_of_b_in_geometric_sequence_l108_10860

theorem min_value_of_b_in_geometric_sequence (a b c : ℝ) : 
  (∃ r : ℝ, (a = b / r ∧ c = b * r) ∨ (a = b * r ∧ c = b / r)) →  -- geometric sequence condition
  ((a = 1 ∧ c = 4) ∨ (a = 4 ∧ c = 1) ∨ (a = 1 ∧ b = 4) ∨ (a = 4 ∧ b = 1) ∨ (b = 1 ∧ c = 4) ∨ (b = 4 ∧ c = 1)) →  -- 1 and 4 are in the sequence
  b ≥ -2 ∧ ∃ b₀ : ℝ, b₀ = -2 ∧ 
    (∃ r : ℝ, (b₀ = b₀ / r ∧ 4 = b₀ * r) ∨ (1 = b₀ * r ∧ 4 = b₀ / r)) ∧
    ((1 = 1 ∧ 4 = 4) ∨ (1 = 4 ∧ 4 = 1) ∨ (1 = 1 ∧ b₀ = 4) ∨ (1 = 4 ∧ b₀ = 1) ∨ (b₀ = 1 ∧ 4 = 4) ∨ (b₀ = 4 ∧ 4 = 1)) :=
by
  sorry

end min_value_of_b_in_geometric_sequence_l108_10860


namespace ellipse_condition_l108_10823

/-- The equation of an ellipse -/
def ellipse_equation (x y b : ℝ) : Prop :=
  4 * x^2 + 9 * y^2 - 16 * x + 18 * y + 12 = b

/-- A non-degenerate ellipse condition -/
def is_non_degenerate_ellipse (b : ℝ) : Prop :=
  b > -13

/-- Theorem: The given equation represents a non-degenerate ellipse iff b > -13 -/
theorem ellipse_condition (b : ℝ) :
  (∃ x y : ℝ, ellipse_equation x y b) ↔ is_non_degenerate_ellipse b :=
sorry

end ellipse_condition_l108_10823


namespace arithmetic_sequence_general_term_l108_10833

/-- An arithmetic sequence with sum formula S_n = n^2 - 3n has general term a_n = 2n - 4 -/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_sum : ∀ n : ℕ, S n = n^2 - 3*n)
  (h_arithmetic : ∀ n : ℕ, a (n+1) - a n = a (n+2) - a (n+1))
  (h_relation : ∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1))
  : ∀ n : ℕ, a n = 2*n - 4 :=
by sorry

end arithmetic_sequence_general_term_l108_10833


namespace exam_mistakes_l108_10887

theorem exam_mistakes (total_students : ℕ) (total_mistakes : ℕ) 
  (h1 : total_students = 333) (h2 : total_mistakes = 1000) : 
  ∀ (x y z : ℕ), 
    (x + y + z = total_students) → 
    (4 * y + 6 * z ≤ total_mistakes) → 
    (z ≤ x) :=
by
  sorry

end exam_mistakes_l108_10887


namespace constant_triangle_area_l108_10895

noncomputable section

-- Define the curve C: xy = 1, x > 0
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 = 1 ∧ p.1 > 0}

-- Define a point P on the curve C
def P (a : ℝ) : ℝ × ℝ := (a, 1/a)

-- Define the tangent line l at point P
def tangent_line (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - 1/a = -(1/a^2) * (p.1 - a)}

-- Define points A and B as intersections of tangent line with axes
def A (a : ℝ) : ℝ × ℝ := (0, 2/a)
def B (a : ℝ) : ℝ × ℝ := (2*a, 0)

-- Define the area of triangle OAB
def triangle_area (a : ℝ) : ℝ := (1/2) * (2/a) * (2*a)

-- Theorem statement
theorem constant_triangle_area (a : ℝ) (h : a > 0) :
  P a ∈ C → triangle_area a = 2 := by sorry

end

end constant_triangle_area_l108_10895


namespace negation_truth_values_l108_10818

theorem negation_truth_values :
  (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ∧
  (¬ ∀ x y : ℝ, Real.sqrt ((x - 1)^2) + (y + 1)^2 ≠ 0) :=
by sorry

end negation_truth_values_l108_10818


namespace skirt_width_is_four_l108_10817

-- Define the width of the rectangle for each skirt
def width : ℝ := sorry

-- Define the length of the rectangle for each skirt
def length : ℝ := 12

-- Define the number of skirts
def num_skirts : ℕ := 3

-- Define the area of material for the bodice
def bodice_area : ℝ := 2 + 2 * 5

-- Define the cost per square foot of material
def cost_per_sqft : ℝ := 3

-- Define the total cost of material
def total_cost : ℝ := 468

-- Theorem statement
theorem skirt_width_is_four :
  width = 4 ∧
  length * width * num_skirts + bodice_area = total_cost / cost_per_sqft :=
sorry

end skirt_width_is_four_l108_10817


namespace mary_bill_difference_l108_10870

/-- Represents the candy distribution problem --/
def candy_distribution (total : ℕ) (kate robert mary bill : ℕ) : Prop :=
  total = 20 ∧
  robert = kate + 2 ∧
  mary > bill ∧
  mary = robert + 2 ∧
  kate = bill + 2 ∧
  kate = 4

/-- Theorem stating the difference between Mary's and Bill's candy pieces --/
theorem mary_bill_difference (total kate robert mary bill : ℕ) 
  (h : candy_distribution total kate robert mary bill) : 
  mary - bill = 6 := by sorry

end mary_bill_difference_l108_10870


namespace cubic_quadratic_comparison_quadratic_inequality_l108_10855

-- Problem 1
theorem cubic_quadratic_comparison (x : ℝ) (h : x ≥ -1) :
  x^3 + 1 ≥ x^2 + x ∧ (x^3 + 1 = x^2 + x ↔ x = 1 ∨ x = -1) := by sorry

-- Problem 2
theorem quadratic_inequality (a x : ℝ) (h : a < 0) :
  x^2 - a*x - 6*a^2 > 0 ↔ x < 3*a ∨ x > -2*a := by sorry

end cubic_quadratic_comparison_quadratic_inequality_l108_10855


namespace negation_of_existence_proposition_l108_10807

theorem negation_of_existence_proposition :
  (¬ ∃ n : ℕ, n^2 ≥ 2^n) ↔ (∀ n : ℕ, n^2 < 2^n) := by
  sorry

end negation_of_existence_proposition_l108_10807


namespace two_evaluations_determine_sequence_l108_10815

/-- A finite sequence of natural numbers -/
def Sequence := List Nat

/-- Evaluate the polynomial at a given point -/
def evaluatePolynomial (s : Sequence) (β : Nat) : Nat :=
  s.enum.foldl (fun acc (i, a) => acc + a * β ^ i) 0

/-- Theorem stating that two evaluations are sufficient to determine the sequence -/
theorem two_evaluations_determine_sequence (s : Sequence) :
  ∃ β₁ β₂ : Nat, β₁ ≠ β₂ ∧
  ∀ t : Sequence, t.length = s.length →
    evaluatePolynomial s β₁ = evaluatePolynomial t β₁ ∧
    evaluatePolynomial s β₂ = evaluatePolynomial t β₂ →
    s = t :=
  sorry

end two_evaluations_determine_sequence_l108_10815


namespace factorization_x3_minus_9xy2_l108_10834

theorem factorization_x3_minus_9xy2 (x y : ℝ) : 
  x^3 - 9*x*y^2 = x*(x+3*y)*(x-3*y) := by sorry

end factorization_x3_minus_9xy2_l108_10834


namespace no_natural_solution_l108_10883

theorem no_natural_solution : ¬∃ (x y : ℕ), x^4 - y^4 = x^3 + y^3 := by
  sorry

end no_natural_solution_l108_10883


namespace range_of_b_l108_10830

theorem range_of_b (a b c : ℝ) (h1 : a * c = b^2) (h2 : a + b + c = 3) :
  -3 ≤ b ∧ b ≤ 1 :=
by sorry

end range_of_b_l108_10830


namespace car_distance_traveled_l108_10847

-- Define constants
def tire_diameter : ℝ := 15
def revolutions : ℝ := 672.1628045157456
def inches_per_mile : ℝ := 63360

-- Define the theorem
theorem car_distance_traveled (ε : ℝ) (h_ε : ε > 0) :
  ∃ (distance : ℝ), 
    abs (distance - 0.5) < ε ∧ 
    distance = (π * tire_diameter * revolutions) / inches_per_mile :=
sorry

end car_distance_traveled_l108_10847


namespace kitchen_guest_bath_living_area_l108_10898

/-- Calculates the area of the kitchen, guest bath, and living area given the areas of other rooms and rent information -/
theorem kitchen_guest_bath_living_area 
  (master_bath_area : ℝ) 
  (guest_bedroom_area : ℝ) 
  (num_guest_bedrooms : ℕ) 
  (total_rent : ℝ) 
  (cost_per_sqft : ℝ) 
  (h1 : master_bath_area = 500) 
  (h2 : guest_bedroom_area = 200) 
  (h3 : num_guest_bedrooms = 2) 
  (h4 : total_rent = 3000) 
  (h5 : cost_per_sqft = 2) : 
  ℝ := by
  sorry

#check kitchen_guest_bath_living_area

end kitchen_guest_bath_living_area_l108_10898


namespace equation_equals_twentyfour_l108_10809

theorem equation_equals_twentyfour : 6 / (1 - 3 / 10) = 24 := by sorry

end equation_equals_twentyfour_l108_10809


namespace meeting_point_theorem_l108_10877

/-- Represents a point on the perimeter of the block area -/
structure Point where
  distance : ℝ  -- Distance from the starting point A
  mk_point_valid : 0 ≤ distance ∧ distance < 24

/-- Represents a walker -/
structure Walker where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The scenario of the two walkers -/
def scenario : Prop :=
  ∃ (jane hector : Walker) (meeting_point : Point),
    jane.speed = 2 * hector.speed ∧
    jane.direction ≠ hector.direction ∧
    (jane.direction = true → meeting_point.distance = 16) ∧
    (jane.direction = false → meeting_point.distance = 8)

/-- The theorem to be proved -/
theorem meeting_point_theorem :
  scenario → 
  ∃ (meeting_point : Point), 
    (meeting_point.distance = 8 ∨ meeting_point.distance = 16) :=
by
  sorry

end meeting_point_theorem_l108_10877


namespace notebook_sales_plan_exists_l108_10857

/-- Represents the sales data for a month -/
structure MonthSales where
  price : ℝ
  sales : ℝ

/-- Represents the notebook sales problem -/
structure NotebookSales where
  initial_inventory : ℕ
  purchase_price : ℝ
  min_sell_price : ℝ
  max_sell_price : ℝ
  july_oct_sales : List MonthSales
  price_sales_relation : ℝ → ℝ

/-- Represents a pricing plan for November and December -/
structure PricingPlan where
  nov_price : ℝ
  nov_sales : ℝ
  dec_price : ℝ
  dec_sales : ℝ

/-- Main theorem statement -/
theorem notebook_sales_plan_exists (problem : NotebookSales) :
  problem.initial_inventory = 550 ∧
  problem.purchase_price = 6 ∧
  problem.min_sell_price = 9 ∧
  problem.max_sell_price = 12 ∧
  problem.july_oct_sales = [⟨9, 115⟩, ⟨10, 100⟩, ⟨11, 85⟩, ⟨12, 70⟩] ∧
  (∀ x, problem.price_sales_relation x = -15 * x + 250) →
  ∃ (plan : PricingPlan),
    -- Remaining inventory after 4 months is 180
    (problem.initial_inventory - (problem.july_oct_sales.map (λ s => s.sales)).sum = 180) ∧
    -- Highest monthly profit in first 4 months is 425, occurring in September
    ((problem.july_oct_sales.map (λ s => (s.price - problem.purchase_price) * s.sales)).maximum = some 425) ∧
    -- Total sales profit for November and December is at least 800
    ((plan.nov_price - problem.purchase_price) * plan.nov_sales +
     (plan.dec_price - problem.purchase_price) * plan.dec_sales ≥ 800) ∧
    -- Pricing plan follows the price-sales relationship
    (problem.price_sales_relation plan.nov_price = plan.nov_sales ∧
     problem.price_sales_relation plan.dec_price = plan.dec_sales) ∧
    -- Prices are within the allowed range
    (plan.nov_price ≥ problem.min_sell_price ∧ plan.nov_price ≤ problem.max_sell_price ∧
     plan.dec_price ≥ problem.min_sell_price ∧ plan.dec_price ≤ problem.max_sell_price) :=
by sorry


end notebook_sales_plan_exists_l108_10857


namespace sum_of_digits_equation_l108_10825

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_equation : 
  ∃ x : ℕ, x + S x = 2001 ∧ x = 1977 := by sorry

end sum_of_digits_equation_l108_10825


namespace inequality_proof_l108_10885

theorem inequality_proof (p : ℝ) (hp : p > 1) :
  ∃ (K_p : ℝ), K_p > 0 ∧
  ∀ (x y : ℝ), (|x|^p + |y|^p = 2) →
  (x - y)^2 ≤ K_p * (4 - (x + y)^2) :=
by sorry

end inequality_proof_l108_10885


namespace right_triangle_hypotenuse_l108_10814

/-- Given a right triangle, prove that if rotation about one leg produces a cone
    of volume 1620π cm³ and rotation about the other leg produces a cone
    of volume 3240π cm³, then the length of the hypotenuse is √507 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1 / 3 * π * a * b^2 = 1620 * π) →
  (1 / 3 * π * b * a^2 = 3240 * π) →
  Real.sqrt (a^2 + b^2) = Real.sqrt 507 := by
  sorry

end right_triangle_hypotenuse_l108_10814


namespace interest_rate_calculation_l108_10837

/-- Proves that the interest rate at which A lends to B is 8% per annum -/
theorem interest_rate_calculation (principal : ℝ) (rate_C : ℝ) (time : ℝ) (gain_B : ℝ)
  (h1 : principal = 3150)
  (h2 : rate_C = 12.5)
  (h3 : time = 2)
  (h4 : gain_B = 283.5) :
  let interest_C := principal * rate_C / 100 * time
  let rate_A := (interest_C - gain_B) / (principal * time) * 100
  rate_A = 8 := by sorry

end interest_rate_calculation_l108_10837


namespace ellipse_focus_implies_k_l108_10896

/-- Represents an ellipse with equation kx^2 + 5y^2 = 5 -/
structure Ellipse (k : ℝ) where
  equation : ∀ x y : ℝ, k * x^2 + 5 * y^2 = 5

/-- A focus of an ellipse -/
def Focus := ℝ × ℝ

/-- Theorem: For the ellipse kx^2 + 5y^2 = 5, if one of its foci is (2, 0), then k = 1 -/
theorem ellipse_focus_implies_k (k : ℝ) (e : Ellipse k) (f : Focus) :
  f = (2, 0) → k = 1 := by
  sorry

end ellipse_focus_implies_k_l108_10896


namespace custom_op_equality_l108_10820

/-- Custom binary operation on real numbers -/
def custom_op (a b : ℝ) : ℝ := a * b + a - b

/-- Theorem stating the equality for the given expression -/
theorem custom_op_equality (a b : ℝ) :
  custom_op a b + custom_op (b - a) b = b^2 - b :=
by sorry

end custom_op_equality_l108_10820


namespace work_completion_time_l108_10871

theorem work_completion_time 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (joint_work_days : ℕ) 
  (a_rate_def : a_rate = 1 / 5) 
  (b_rate_def : b_rate = 1 / 15) 
  (joint_work_days_def : joint_work_days = 2) : 
  (1 - (a_rate + b_rate) * joint_work_days) / b_rate = 7 := by
  sorry

end work_completion_time_l108_10871


namespace time_before_first_rewind_is_35_l108_10810

/-- Represents the viewing time of a movie with interruptions -/
structure MovieViewing where
  totalTime : ℕ
  firstRewindTime : ℕ
  timeBetweenRewinds : ℕ
  secondRewindTime : ℕ
  timeAfterSecondRewind : ℕ

/-- Calculates the time watched before the first rewind -/
def timeBeforeFirstRewind (mv : MovieViewing) : ℕ :=
  mv.totalTime - (mv.firstRewindTime + mv.timeBetweenRewinds + mv.secondRewindTime + mv.timeAfterSecondRewind)

/-- Theorem stating that for the given movie viewing scenario, 
    the time watched before the first rewind is 35 minutes -/
theorem time_before_first_rewind_is_35 : 
  let mv : MovieViewing := {
    totalTime := 120,
    firstRewindTime := 5,
    timeBetweenRewinds := 45,
    secondRewindTime := 15,
    timeAfterSecondRewind := 20
  }
  timeBeforeFirstRewind mv = 35 := by
  sorry

end time_before_first_rewind_is_35_l108_10810


namespace sqrt_two_expansion_l108_10801

theorem sqrt_two_expansion (a b : ℚ) : 
  (1 + Real.sqrt 2)^5 = a + Real.sqrt 2 * b → a - b = 12 := by
  sorry

end sqrt_two_expansion_l108_10801


namespace seed_germination_percentage_l108_10873

/-- Given an agricultural experiment with two plots of seeds, calculate the percentage of total seeds that germinated. -/
theorem seed_germination_percentage 
  (seeds_plot1 : ℕ) 
  (seeds_plot2 : ℕ) 
  (germination_rate_plot1 : ℚ) 
  (germination_rate_plot2 : ℚ) 
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 25 / 100)
  (h4 : germination_rate_plot2 = 40 / 100) :
  (((seeds_plot1 : ℚ) * germination_rate_plot1 + (seeds_plot2 : ℚ) * germination_rate_plot2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) = 31 / 100 := by
  sorry

end seed_germination_percentage_l108_10873


namespace smallest_three_digit_multiple_of_17_l108_10852

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end smallest_three_digit_multiple_of_17_l108_10852


namespace order_of_numbers_l108_10802

theorem order_of_numbers : 
  0 < 0.89 → 0.89 < 1 → 90.8 > 1 → Real.log 0.89 < 0 → 
  Real.log 0.89 < 0.89 ∧ 0.89 < 90.8 := by
  sorry

end order_of_numbers_l108_10802


namespace product_zero_l108_10881

theorem product_zero (b : ℤ) (h : b = 3) : 
  (b - 13) * (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * 
  (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by sorry

end product_zero_l108_10881


namespace tom_found_seven_seashells_l108_10800

/-- The number of seashells Tom found yesterday -/
def seashells_yesterday : ℕ := sorry

/-- The number of seashells Tom found today -/
def seashells_today : ℕ := 4

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := 11

/-- Theorem stating that Tom found 7 seashells yesterday -/
theorem tom_found_seven_seashells : seashells_yesterday = 7 := by
  sorry

end tom_found_seven_seashells_l108_10800


namespace series_evaluation_l108_10822

noncomputable def series_sum : ℝ := ∑' k, (k : ℝ) / (4 ^ k)

theorem series_evaluation : series_sum = 4 / 9 := by
  sorry

end series_evaluation_l108_10822


namespace polynomial_division_remainder_l108_10899

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ,
  2 * X^5 + 11 * X^4 - 48 * X^3 - 60 * X^2 + 20 * X + 50 =
  (X^3 + 7 * X^2 + 4) * q + (-27 * X^3 - 68 * X^2 + 32 * X + 50) := by
  sorry

end polynomial_division_remainder_l108_10899


namespace bike_price_calculation_l108_10893

theorem bike_price_calculation (current_price : ℝ) : 
  (current_price * 1.1 = 82500) → current_price = 75000 := by
  sorry

end bike_price_calculation_l108_10893


namespace sum_of_powers_l108_10875

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^10 + b^10 = 123 -/
theorem sum_of_powers (a b : ℝ) 
  (h1 : a = Real.sqrt 6)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^4 + b^4 = 7)
  (h4 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := by
  sorry

end sum_of_powers_l108_10875


namespace track_length_satisfies_conditions_l108_10868

/-- The length of a circular track satisfying the given conditions -/
def track_length : ℝ := 766.67

/-- Two runners on a circular track -/
structure Runners :=
  (track_length : ℝ)
  (initial_separation : ℝ)
  (first_meeting_distance : ℝ)
  (second_meeting_distance : ℝ)

/-- The conditions of the problem -/
def problem_conditions (r : Runners) : Prop :=
  r.initial_separation = 0.75 * r.track_length ∧
  r.first_meeting_distance = 120 ∧
  r.second_meeting_distance = 180

/-- The theorem stating that the track length satisfies the problem conditions -/
theorem track_length_satisfies_conditions :
  ∃ (r : Runners), r.track_length = track_length ∧ problem_conditions r :=
sorry

end track_length_satisfies_conditions_l108_10868


namespace tax_reduction_is_sixty_percent_tax_reduction_for_thousand_million_l108_10805

/-- Represents the tax reduction process for a commodity -/
def tax_reduction (initial_tax : ℝ) : ℝ :=
  let after_first_reduction := initial_tax * (1 - 0.25)
  let after_second_reduction := after_first_reduction * (1 - 0.20)
  after_second_reduction

/-- Theorem stating that the tax reduction process results in 60% of the initial tax -/
theorem tax_reduction_is_sixty_percent (a : ℝ) :
  tax_reduction a = 0.60 * a := by
  sorry

/-- Corollary for the specific case where the initial tax is 1000 million euros -/
theorem tax_reduction_for_thousand_million :
  tax_reduction 1000 = 600 := by
  sorry

end tax_reduction_is_sixty_percent_tax_reduction_for_thousand_million_l108_10805


namespace no_triangle_tangent_and_inscribed_l108_10804

/-- The problem statement as a theorem -/
theorem no_triangle_tangent_and_inscribed (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let C₂ : Set (ℝ × ℝ) := {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  (1 : ℝ)^2 / a^2 + (1 : ℝ)^2 / b^2 = 1 →
  ¬ ∃ (A B C : ℝ × ℝ),
    (A ∈ C₂ ∧ B ∈ C₂ ∧ C ∈ C₂) ∧
    (∀ p : ℝ × ℝ, p ∈ C₁ → (dist p A ≥ dist A B ∧ dist p B ≥ dist A B ∧ dist p C ≥ dist A B)) :=
by
  sorry


end no_triangle_tangent_and_inscribed_l108_10804


namespace max_value_theorem_max_value_achievable_l108_10872

theorem max_value_theorem (x y : ℝ) :
  (3 * x + 4 * y + 6) / Real.sqrt (x^2 + 4 * y^2 + 4) ≤ Real.sqrt 61 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (3 * x + 4 * y + 6) / Real.sqrt (x^2 + 4 * y^2 + 4) = Real.sqrt 61 :=
by sorry

end max_value_theorem_max_value_achievable_l108_10872


namespace pencil_count_theorem_pencils_in_drawer_l108_10842

/-- The total number of pencils after adding more to the drawer -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of pencils is the sum of initial and added pencils -/
theorem pencil_count_theorem (initial : ℕ) (added : ℕ) :
  total_pencils initial added = initial + added := by
  sorry

/-- Given information about the pencils in the drawer -/
def initial_pencils : ℕ := 41
def pencils_added : ℕ := 30

/-- The result we want to prove -/
theorem pencils_in_drawer :
  total_pencils initial_pencils pencils_added = 71 := by
  sorry

end pencil_count_theorem_pencils_in_drawer_l108_10842


namespace rice_bag_weight_l108_10835

theorem rice_bag_weight (rice_bags : ℕ) (flour_bags : ℕ) (total_weight : ℕ) :
  rice_bags = 20 →
  flour_bags = 50 →
  total_weight = 2250 →
  (∃ (rice_weight flour_weight : ℕ),
    rice_weight * rice_bags + flour_weight * flour_bags = total_weight ∧
    rice_weight = 2 * flour_weight) →
  ∃ (rice_weight : ℕ), rice_weight = 50 :=
by sorry

end rice_bag_weight_l108_10835


namespace differential_equation_satisfaction_l108_10853

open Real

theorem differential_equation_satisfaction (n : ℝ) (x : ℝ) (h : x ≠ -1) :
  let y : ℝ → ℝ := λ x => (x + 1)^n * (exp x - 1)
  deriv y x - (n * y x) / (x + 1) = exp x * (1 + x)^n := by
  sorry

end differential_equation_satisfaction_l108_10853


namespace min_period_sin_2x_cos_2x_l108_10816

/-- The minimum positive period of the function y = sin(2x) cos(2x) is π/2 -/
theorem min_period_sin_2x_cos_2x :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x) * Real.cos (2 * x)
  ∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi / 2 :=
by sorry

end min_period_sin_2x_cos_2x_l108_10816


namespace river_length_estimate_l108_10851

/-- Represents a measurement with an associated error probability -/
structure Measurement where
  value : ℝ
  error : ℝ
  errorProb : ℝ

/-- Calculates the best estimate and error probability given two measurements -/
def calculateEstimate (m1 m2 : Measurement) : ℝ × ℝ :=
  sorry

theorem river_length_estimate 
  (gsa awra : Measurement)
  (h1 : gsa.value = 402)
  (h2 : gsa.error = 0.5)
  (h3 : gsa.errorProb = 0.04)
  (h4 : awra.value = 403)
  (h5 : awra.error = 0.5)
  (h6 : awra.errorProb = 0.04) :
  calculateEstimate gsa awra = (402.5, 0.04) :=
sorry

end river_length_estimate_l108_10851


namespace strawberry_jelly_amount_l108_10856

/-- Given the total amount of jelly and the amount of blueberry jelly, 
    calculate the amount of strawberry jelly. -/
theorem strawberry_jelly_amount 
  (total_jelly : ℕ) 
  (blueberry_jelly : ℕ) 
  (h1 : total_jelly = 6310)
  (h2 : blueberry_jelly = 4518) : 
  total_jelly - blueberry_jelly = 1792 := by
sorry

end strawberry_jelly_amount_l108_10856


namespace sum_of_squares_over_products_l108_10840

theorem sum_of_squares_over_products (a b c : ℝ) (h : a + b + c = 0) :
  a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 3 :=
by sorry

end sum_of_squares_over_products_l108_10840


namespace area_of_pentagon_l108_10839

-- Define the square ABCD
def ABCD : Set (ℝ × ℝ) := sorry

-- Define that BD is a diagonal of ABCD
def BD_is_diagonal (ABCD : Set (ℝ × ℝ)) : Prop := sorry

-- Define the length of BD
def BD_length : ℝ := 20

-- Define the rectangle BDFE
def BDFE : Set (ℝ × ℝ) := sorry

-- Define the pentagon ABEFD
def ABEFD : Set (ℝ × ℝ) := sorry

-- Define the area function
def area : Set (ℝ × ℝ) → ℝ := sorry

-- Theorem statement
theorem area_of_pentagon (h1 : BD_is_diagonal ABCD) (h2 : BD_length = 20) : 
  area ABEFD = 300 := by sorry

end area_of_pentagon_l108_10839


namespace hexagon_area_in_circle_l108_10874

/-- The area of a regular hexagon inscribed in a circle with radius 2 units is 6√3 square units. -/
theorem hexagon_area_in_circle (r : ℝ) (h : r = 2) : 
  let hexagon_area := 6 * (r^2 * Real.sqrt 3 / 4)
  hexagon_area = 6 * Real.sqrt 3 := by
  sorry

end hexagon_area_in_circle_l108_10874


namespace number_symmetry_equation_l108_10869

theorem number_symmetry_equation (a b : ℕ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10 * a + b) * (100 * b + 10 * (a + b) + a) = (100 * a + 10 * (a + b) + b) * (10 * b + a) := by
  sorry

end number_symmetry_equation_l108_10869


namespace cleaning_payment_l108_10850

theorem cleaning_payment (payment_rate rooms_cleaned : ℚ) : 
  payment_rate = 13 / 3 → rooms_cleaned = 8 / 5 → payment_rate * rooms_cleaned = 104 / 15 := by
  sorry

end cleaning_payment_l108_10850


namespace negation_of_absolute_value_nonnegative_l108_10854

theorem negation_of_absolute_value_nonnegative :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by sorry

end negation_of_absolute_value_nonnegative_l108_10854


namespace sculpture_cost_in_yen_l108_10859

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Exchange rate from US dollars to Japanese yen -/
def usd_to_jpy : ℚ := 110

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 136

/-- Theorem stating the cost of the sculpture in Japanese yen -/
theorem sculpture_cost_in_yen : 
  (sculpture_cost_nad / usd_to_nad) * usd_to_jpy = 1870 := by
  sorry

end sculpture_cost_in_yen_l108_10859


namespace tangent_lines_to_cubic_l108_10863

noncomputable def f (x : ℝ) := x^3

def P : ℝ × ℝ := (1, 1)

theorem tangent_lines_to_cubic (x : ℝ) :
  -- The tangent line at point P(1, 1) is y = 3x - 2
  (HasDerivAt f 3 1 ∧ f 1 = 1) →
  -- There are exactly two tangent lines to the curve that pass through point P(1, 1)
  (∃! (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧
    -- First tangent line: y = 3x - 2
    (m₁ = 3 ∧ P.2 = m₁ * P.1 - 2) ∧
    -- Second tangent line: y = 3/4x + 1/4
    (m₂ = 3/4 ∧ P.2 = m₂ * P.1 + 1/4) ∧
    -- Both lines are tangent to the curve
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
      HasDerivAt f (3 * x₁^2) x₁ ∧
      HasDerivAt f (3 * x₂^2) x₂ ∧
      f x₁ = m₁ * x₁ - 2 ∧
      f x₂ = m₂ * x₂ + 1/4)) :=
by sorry

end tangent_lines_to_cubic_l108_10863


namespace ecu_distribution_l108_10886

theorem ecu_distribution (x y z : ℤ) : 
  (x - y - z = 8) ∧ 
  (y - (x - y - z) - z = 8) ∧ 
  (z - (x - y - z) - (y - (x - y - z)) = 8) → 
  x = 13 ∧ y = 7 ∧ z = 4 := by
sorry

end ecu_distribution_l108_10886


namespace profit_difference_l108_10864

/-- Calculates the difference in profit share between two partners given their investments and total profit -/
theorem profit_difference (mary_investment mike_investment total_profit : ℚ) :
  mary_investment = 650 →
  mike_investment = 350 →
  total_profit = 2999.9999999999995 →
  let total_investment := mary_investment + mike_investment
  let equal_share := (1/3) * total_profit / 2
  let remaining_profit := (2/3) * total_profit
  let mary_ratio := mary_investment / total_investment
  let mike_ratio := mike_investment / total_investment
  let mary_total := equal_share + mary_ratio * remaining_profit
  let mike_total := equal_share + mike_ratio * remaining_profit
  mary_total - mike_total = 600 := by sorry

end profit_difference_l108_10864


namespace conference_beverages_l108_10876

theorem conference_beverages (total participants : ℕ) (coffee_drinkers : ℕ) (tea_drinkers : ℕ) (both_drinkers : ℕ) :
  total = 30 →
  coffee_drinkers = 15 →
  tea_drinkers = 18 →
  both_drinkers = 8 →
  total - (coffee_drinkers + tea_drinkers - both_drinkers) = 5 := by
sorry

end conference_beverages_l108_10876


namespace monotonic_function_implies_a_le_e_l108_10838

/-- Given f(x) = 2xe^x - ax^2 - 2ax is monotonically increasing on [1, +∞), prove that a ≤ e -/
theorem monotonic_function_implies_a_le_e (a : ℝ) :
  (∀ x ≥ 1, Monotone (fun x => 2 * x * Real.exp x - a * x^2 - 2 * a * x)) →
  a ≤ Real.exp 1 :=
by sorry

end monotonic_function_implies_a_le_e_l108_10838


namespace x_percent_plus_six_equals_ten_l108_10829

theorem x_percent_plus_six_equals_ten (x : ℝ) (h1 : x > 0) 
  (h2 : x * (x / 100) + 6 = 10) : x = 20 := by
  sorry

end x_percent_plus_six_equals_ten_l108_10829


namespace milk_mixture_problem_l108_10836

/-- Proves that the volume removed and replaced with water is 50 litres -/
theorem milk_mixture_problem (total_volume : ℝ) (initial_milk : ℝ) (final_concentration : ℝ) :
  total_volume = 100 →
  initial_milk = 36 →
  final_concentration = 0.09 →
  ∃ (V : ℝ), V = 50 ∧
    (initial_milk / total_volume) * (1 - V / total_volume)^2 = final_concentration :=
by sorry

end milk_mixture_problem_l108_10836


namespace longest_chord_in_quarter_circle_l108_10865

theorem longest_chord_in_quarter_circle (d : ℝ) (h : d = 16) : 
  let r := d / 2
  let chord_length := (2 * r ^ 2) ^ (1/2)
  chord_length ^ 2 = 128 :=
by sorry

end longest_chord_in_quarter_circle_l108_10865


namespace negative_fractions_comparison_l108_10811

theorem negative_fractions_comparison : -4/5 < -2/3 := by
  sorry

end negative_fractions_comparison_l108_10811


namespace greatest_number_l108_10832

theorem greatest_number (x : ℤ) (n : ℕ) : 
  (x ≤ 4) → 
  (2.134 * (n : ℝ)^(x : ℝ) < 210000) → 
  (∀ m : ℕ, m > n → 2.134 * (m : ℝ)^(4 : ℝ) ≥ 210000) → 
  n = 17 := by
sorry

end greatest_number_l108_10832


namespace least_positive_even_congruence_l108_10808

theorem least_positive_even_congruence : ∃ x : ℕ, 
  (x + 3721 : ℤ) ≡ 1547 [ZMOD 12] ∧ 
  Even x ∧
  x = 2 ∧
  ∀ y : ℕ, y < x → ¬((y + 3721 : ℤ) ≡ 1547 [ZMOD 12] ∧ Even y) := by
  sorry

end least_positive_even_congruence_l108_10808


namespace units_digit_F_500_l108_10813

-- Define the modified Fermat number function
def F (n : ℕ) : ℕ := 2^(2^(2*n)) + 1

-- Theorem statement
theorem units_digit_F_500 : F 500 % 10 = 7 := by sorry

end units_digit_F_500_l108_10813


namespace dividend_calculation_l108_10882

theorem dividend_calculation (divisor quotient remainder : ℕ) :
  divisor = 15 →
  quotient = 8 →
  remainder = 5 →
  (divisor * quotient + remainder) = 125 := by
  sorry

end dividend_calculation_l108_10882


namespace fries_ratio_l108_10821

def sally_initial_fries : ℕ := 14
def mark_initial_fries : ℕ := 36
def sally_final_fries : ℕ := 26

def fries_mark_gave_sally : ℕ := sally_final_fries - sally_initial_fries

theorem fries_ratio :
  (fries_mark_gave_sally : ℚ) / mark_initial_fries = 1 / 3 := by
  sorry

end fries_ratio_l108_10821


namespace first_business_donation_is_half_dollar_l108_10828

/-- Represents the fundraising scenario for Didi's soup kitchen --/
structure FundraisingScenario where
  num_cakes : ℕ
  slices_per_cake : ℕ
  price_per_slice : ℚ
  second_business_donation : ℚ
  total_raised : ℚ

/-- Calculates the donation per slice from the first business owner --/
def first_business_donation_per_slice (scenario : FundraisingScenario) : ℚ :=
  let total_slices := scenario.num_cakes * scenario.slices_per_cake
  let sales_revenue := total_slices * scenario.price_per_slice
  let total_business_donations := scenario.total_raised - sales_revenue
  let second_business_total := total_slices * scenario.second_business_donation
  let first_business_total := total_business_donations - second_business_total
  first_business_total / total_slices

/-- Theorem stating that the first business owner's donation per slice is $0.50 --/
theorem first_business_donation_is_half_dollar (scenario : FundraisingScenario) 
  (h1 : scenario.num_cakes = 10)
  (h2 : scenario.slices_per_cake = 8)
  (h3 : scenario.price_per_slice = 1)
  (h4 : scenario.second_business_donation = 1/4)
  (h5 : scenario.total_raised = 140) :
  first_business_donation_per_slice scenario = 1/2 := by
  sorry


end first_business_donation_is_half_dollar_l108_10828


namespace arithmetic_sequence_length_l108_10845

theorem arithmetic_sequence_length (first last step : ℤ) (h : first ≥ last) : 
  (first - last) / step + 1 = (first - 44) / 4 + 1 → (first - 44) / 4 + 1 = 28 :=
by
  sorry

end arithmetic_sequence_length_l108_10845


namespace student_selection_probability_l108_10892

/-- Given a set of 3 students where 2 are to be selected, 
    the probability of a specific student being selected is 2/3 -/
theorem student_selection_probability 
  (S : Finset Nat) 
  (h_card : S.card = 3) 
  (A : Nat) 
  (h_A_in_S : A ∈ S) : 
  Nat.card {pair : Finset Nat | pair ⊆ S ∧ pair.card = 2 ∧ A ∈ pair} / 
  Nat.card {pair : Finset Nat | pair ⊆ S ∧ pair.card = 2} = 2 / 3 := by
  sorry

end student_selection_probability_l108_10892


namespace real_part_of_complex_product_l108_10894

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (1 - Complex.I) * (2 + 3 * Complex.I) ∧ z.re = 5 := by
  sorry

end real_part_of_complex_product_l108_10894


namespace two_dogs_walking_time_l108_10819

def dog_walking_earnings (base_charge : ℕ) (per_minute_charge : ℕ) 
  (dogs_1 : ℕ) (time_1 : ℕ) (dogs_2 : ℕ) (time_2 : ℕ) 
  (dogs_3 : ℕ) (time_3 : ℕ) : ℕ :=
  (dogs_1 * (base_charge + per_minute_charge * time_1)) +
  (dogs_2 * (base_charge + per_minute_charge * time_2)) +
  (dogs_3 * (base_charge + per_minute_charge * time_3))

theorem two_dogs_walking_time : 
  ∃ (time_2 : ℕ), 
    dog_walking_earnings 20 1 1 10 2 time_2 3 9 = 171 ∧ 
    time_2 = 7 := by
  sorry

end two_dogs_walking_time_l108_10819


namespace mass_of_man_on_boat_l108_10861

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sink_depth * water_density

/-- Theorem stating that the mass of the man is 60 kg under given conditions. -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 3
  let boat_breadth : ℝ := 2
  let sink_depth : ℝ := 0.01
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth sink_depth water_density = 60 := by
sorry


end mass_of_man_on_boat_l108_10861


namespace smallest_enclosing_circle_theorem_l108_10889

/-- The radius of the smallest circle from which a triangle with sides 2, 3, and 4 can be cut out --/
def smallest_enclosing_circle_radius : ℝ := 2

/-- The three sides of the triangle --/
def triangle_sides : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => 4
| _ => 0  -- This line is necessary due to Lean's totality requirement

theorem smallest_enclosing_circle_theorem :
  ∀ r : ℝ, (∀ i : Fin 3, triangle_sides i ≤ 2 * r) → r ≥ smallest_enclosing_circle_radius :=
by sorry

end smallest_enclosing_circle_theorem_l108_10889


namespace junior_basketball_league_bad_teams_l108_10884

/-- Given a total of 18 teams in a junior basketball league, where half are rich,
    and there cannot be 10 teams that are both rich and bad,
    prove that the fraction of bad teams must be less than or equal to 1/2. -/
theorem junior_basketball_league_bad_teams
  (total_teams : ℕ)
  (rich_teams : ℕ)
  (bad_fraction : ℚ)
  (h1 : total_teams = 18)
  (h2 : rich_teams = total_teams / 2)
  (h3 : ¬(bad_fraction * ↑total_teams ≥ 10 ∧ bad_fraction * ↑total_teams ≤ ↑rich_teams)) :
  bad_fraction ≤ 1/2 :=
by sorry

end junior_basketball_league_bad_teams_l108_10884


namespace ellipse_slope_product_l108_10844

theorem ellipse_slope_product (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ ≠ 0 →
  y₁ ≠ 0 →
  x₁^2 + 4*y₁^2/9 = 1 →
  x₂^2 + 4*y₂^2/9 = 1 →
  3*y₁/(4*x₁) = (y₁ + y₂)/(x₁ + x₂) →
  (y₁/x₁) * ((y₁ - y₂)/(x₁ - x₂)) = -1 :=
by sorry

end ellipse_slope_product_l108_10844


namespace rogers_second_bag_pieces_l108_10888

/-- Represents the number of candy bags each person has -/
def num_bags : ℕ := 2

/-- Represents the number of candy pieces in each of Sandra's bags -/
def sandra_bag_pieces : ℕ := 6

/-- Represents the number of candy pieces in Roger's first bag -/
def roger_first_bag_pieces : ℕ := 11

/-- Represents the difference in total candy pieces between Roger and Sandra -/
def difference : ℕ := 2

/-- Theorem stating the number of candy pieces in Roger's second bag -/
theorem rogers_second_bag_pieces :
  ∃ (x : ℕ), x = num_bags * sandra_bag_pieces + difference - roger_first_bag_pieces :=
sorry

end rogers_second_bag_pieces_l108_10888


namespace paper_tearing_impossibility_l108_10858

theorem paper_tearing_impossibility : ¬ ∃ (n : ℕ), 1 + 2 * n = 100 := by
  sorry

end paper_tearing_impossibility_l108_10858


namespace problem_solution_l108_10803

theorem problem_solution (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2004 = 2005 := by
  sorry

end problem_solution_l108_10803


namespace divisibility_by_six_l108_10827

theorem divisibility_by_six (m n : ℤ) 
  (h1 : ∃ x y : ℤ, x^2 + m*x - n = 0 ∧ y^2 + m*y - n = 0)
  (h2 : ∃ x y : ℤ, x^2 - m*x + n = 0 ∧ y^2 - m*y + n = 0) : 
  6 ∣ n :=
sorry

end divisibility_by_six_l108_10827


namespace league_games_l108_10879

theorem league_games (num_teams : ℕ) (total_games : ℕ) (games_per_matchup : ℕ) : 
  num_teams = 20 → 
  total_games = 1900 → 
  total_games = (num_teams * (num_teams - 1) * games_per_matchup) / 2 → 
  games_per_matchup = 10 := by
  sorry

end league_games_l108_10879


namespace adams_purchase_cost_l108_10826

/-- The cost of Adam's purchases given the quantities and prices of nuts and dried fruits -/
theorem adams_purchase_cost (nuts_quantity : ℝ) (nuts_price : ℝ) (fruits_quantity : ℝ) (fruits_price : ℝ) 
  (h1 : nuts_quantity = 3)
  (h2 : nuts_price = 12)
  (h3 : fruits_quantity = 2.5)
  (h4 : fruits_price = 8) :
  nuts_quantity * nuts_price + fruits_quantity * fruits_price = 56 := by
  sorry

end adams_purchase_cost_l108_10826


namespace total_distance_rowed_l108_10843

/-- The total distance traveled by a man rowing upstream and downstream -/
theorem total_distance_rowed (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : 
  man_speed = 6 →
  river_speed = 1.2 →
  total_time = 1 →
  2 * (total_time * man_speed * river_speed) / (man_speed + river_speed) = 5.76 :=
by
  sorry

end total_distance_rowed_l108_10843


namespace two_digit_product_digits_l108_10890

theorem two_digit_product_digits (a b : ℕ) (ha : 40 < a ∧ a < 100) (hb : 40 < b ∧ b < 100) :
  (1000 ≤ a * b ∧ a * b < 10000) ∨ (100 ≤ a * b ∧ a * b < 1000) :=
sorry

end two_digit_product_digits_l108_10890


namespace rhombus_side_length_l108_10867

/-- Represents a rhombus with given diagonal and area -/
structure Rhombus where
  diagonal : ℝ
  area : ℝ

/-- Calculates the length of the side of a rhombus -/
def side_length (r : Rhombus) : ℝ :=
  sorry

theorem rhombus_side_length (r : Rhombus) (h1 : r.diagonal = 30) (h2 : r.area = 600) :
  side_length r = 25 := by
  sorry

end rhombus_side_length_l108_10867


namespace positive_expression_l108_10824

theorem positive_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 1) : 
  0 < b + 3 * b^2 := by
  sorry

end positive_expression_l108_10824


namespace two_numbers_with_difference_and_quotient_five_l108_10806

theorem two_numbers_with_difference_and_quotient_five :
  ∀ x y : ℝ, x - y = 5 → x / y = 5 → x = 25 / 4 ∧ y = 5 / 4 := by
  sorry

end two_numbers_with_difference_and_quotient_five_l108_10806


namespace candidate_votes_l108_10878

theorem candidate_votes (total_votes : ℕ) (invalid_percentage : ℚ) (candidate_percentage : ℚ) :
  total_votes = 560000 →
  invalid_percentage = 15 / 100 →
  candidate_percentage = 70 / 100 →
  ∃ (valid_votes : ℕ) (candidate_votes : ℕ),
    valid_votes = (1 - invalid_percentage) * total_votes ∧
    candidate_votes = candidate_percentage * valid_votes ∧
    candidate_votes = 333200 := by
  sorry

end candidate_votes_l108_10878


namespace sequence_property_l108_10897

def sequence_condition (a : ℕ → ℝ) (m r : ℝ) : Prop :=
  a 1 = m ∧
  (∀ k : ℕ, a (2*k) = 2 * a (2*k - 1)) ∧
  (∀ k : ℕ, a (2*k + 1) = a (2*k) + r) ∧
  (∀ n : ℕ, n > 0 → a (n + 2) = a n)

theorem sequence_property (a : ℕ → ℝ) (m r : ℝ) 
  (h : sequence_condition a m r) : m + r = 0 := by
  sorry

end sequence_property_l108_10897
