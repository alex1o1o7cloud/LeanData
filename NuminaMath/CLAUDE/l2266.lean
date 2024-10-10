import Mathlib

namespace box_weight_l2266_226688

/-- Given a pallet with boxes, calculate the weight of each box. -/
theorem box_weight (total_weight : ℝ) (num_boxes : ℕ) (h1 : total_weight = 267) (h2 : num_boxes = 3) :
  total_weight / num_boxes = 89 := by
  sorry

end box_weight_l2266_226688


namespace owen_profit_l2266_226665

/-- Calculate Owen's overall profit from selling face masks --/
theorem owen_profit : 
  let cheap_boxes := 8
  let expensive_boxes := 4
  let cheap_box_price := 9
  let expensive_box_price := 12
  let masks_per_box := 50
  let small_packets := 100
  let small_packet_price := 5
  let small_packet_size := 25
  let large_packets := 28
  let large_packet_price := 12
  let large_packet_size := 100
  let remaining_cheap := 150
  let remaining_expensive := 150
  let remaining_cheap_price := 3
  let remaining_expensive_price := 4

  let total_cost := cheap_boxes * cheap_box_price + expensive_boxes * expensive_box_price
  let total_masks := (cheap_boxes + expensive_boxes) * masks_per_box
  let repacked_revenue := small_packets * small_packet_price + large_packets * large_packet_price
  let remaining_revenue := remaining_cheap * remaining_cheap_price + remaining_expensive * remaining_expensive_price
  let total_revenue := repacked_revenue + remaining_revenue
  let profit := total_revenue - total_cost

  profit = 1766 := by sorry

end owen_profit_l2266_226665


namespace a_range_l2266_226695

/-- The piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x - a

/-- Theorem stating that if f(0) is the minimum value of f(x), then a is in [0,1] --/
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end a_range_l2266_226695


namespace star_wars_earnings_value_l2266_226614

/-- The cost to make The Lion King in millions of dollars -/
def lion_king_cost : ℝ := 10

/-- The box office earnings of The Lion King in millions of dollars -/
def lion_king_earnings : ℝ := 200

/-- The cost to make Star Wars in millions of dollars -/
def star_wars_cost : ℝ := 25

/-- The profit of The Lion King in millions of dollars -/
def lion_king_profit : ℝ := lion_king_earnings - lion_king_cost

/-- The profit of Star Wars in millions of dollars -/
def star_wars_profit : ℝ := 2 * lion_king_profit

/-- The earnings of Star Wars in millions of dollars -/
def star_wars_earnings : ℝ := star_wars_cost + star_wars_profit

theorem star_wars_earnings_value : star_wars_earnings = 405 := by
  sorry

#eval star_wars_earnings

end star_wars_earnings_value_l2266_226614


namespace area_of_AEC_l2266_226671

-- Define the triangle ABC and its area
def triangle_ABC : Real := 40

-- Define the points on the sides of the triangle
def point_D : Real := 3
def point_B : Real := 5

-- Define the equality of areas
def area_equality : Prop := true

-- Theorem to prove
theorem area_of_AEC (triangle_ABC : Real) (point_D point_B : Real) (area_equality : Prop) :
  (3 : Real) / 8 * triangle_ABC = 15 := by
  sorry

end area_of_AEC_l2266_226671


namespace xy_value_l2266_226623

theorem xy_value (x y : ℝ) (h : x * (x - y) = x^2 - 6) : x * y = 6 := by
  sorry

end xy_value_l2266_226623


namespace quadratic_equation_roots_l2266_226680

theorem quadratic_equation_roots (a : ℝ) :
  a = 1 →
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    x₁^2 + (1 - a) * x₁ - 1 = 0 ∧
    x₂^2 + (1 - a) * x₂ - 1 = 0 := by
  sorry

end quadratic_equation_roots_l2266_226680


namespace stream_current_is_three_l2266_226615

/-- Represents the rowing scenario described in the problem -/
structure RowingScenario where
  r : ℝ  -- man's rowing speed in still water (miles per hour)
  c : ℝ  -- speed of the stream's current (miles per hour)
  distance : ℝ  -- distance traveled (miles)
  timeDiffNormal : ℝ  -- time difference between upstream and downstream at normal rate (hours)
  timeDiffTripled : ℝ  -- time difference between upstream and downstream at tripled rate (hours)

/-- The theorem stating that given the problem conditions, the stream's current is 3 mph -/
theorem stream_current_is_three 
  (scenario : RowingScenario)
  (h1 : scenario.distance = 20)
  (h2 : scenario.timeDiffNormal = 6)
  (h3 : scenario.timeDiffTripled = 1.5)
  (h4 : scenario.distance / (scenario.r + scenario.c) + scenario.timeDiffNormal = 
        scenario.distance / (scenario.r - scenario.c))
  (h5 : scenario.distance / (3 * scenario.r + scenario.c) + scenario.timeDiffTripled = 
        scenario.distance / (3 * scenario.r - scenario.c))
  : scenario.c = 3 := by
  sorry

#check stream_current_is_three

end stream_current_is_three_l2266_226615


namespace books_redistribution_l2266_226612

theorem books_redistribution (mark_initial : ℕ) (alice_initial : ℕ) (books_given : ℕ) : 
  mark_initial = 105 →
  alice_initial = 15 →
  books_given = 15 →
  mark_initial - books_given = 3 * (alice_initial + books_given) :=
by
  sorry

end books_redistribution_l2266_226612


namespace floor_ceiling_sum_l2266_226659

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.2 : ℝ)⌉ = 27 := by sorry

end floor_ceiling_sum_l2266_226659


namespace dans_remaining_green_marbles_l2266_226672

def dans_initial_green_marbles : ℕ := 32
def mikes_taken_green_marbles : ℕ := 23

theorem dans_remaining_green_marbles :
  dans_initial_green_marbles - mikes_taken_green_marbles = 9 := by
  sorry

end dans_remaining_green_marbles_l2266_226672


namespace input_statement_separator_l2266_226683

/-- Represents the possible separators in an input statement -/
inductive Separator
  | Comma
  | Space
  | Semicolon
  | Pause

/-- Represents the general format of an input statement -/
structure InputStatement where
  separator : Separator

/-- The correct separator for multiple variables in an input statement -/
def correctSeparator : Separator := Separator.Comma

/-- Theorem stating that the correct separator in the general format of an input statement is a comma -/
theorem input_statement_separator :
  ∀ (stmt : InputStatement), stmt.separator = correctSeparator :=
sorry


end input_statement_separator_l2266_226683


namespace sin_plus_cos_value_l2266_226697

theorem sin_plus_cos_value (x : ℝ) 
  (h1 : 0 < x ∧ x < π/2) 
  (h2 : Real.sin (2*x - π/4) = -Real.sqrt 2/10) : 
  Real.sin x + Real.cos x = 2*Real.sqrt 10/5 := by
sorry

end sin_plus_cos_value_l2266_226697


namespace perfect_square_condition_l2266_226666

/-- The expression ax^2 + 2bxy + cy^2 - k(x^2 + y^2) is a perfect square if and only if 
    k = (a+c)/2 ± (1/2)√((a-c)^2 + 4b^2), where a, b, c are real constants. -/
theorem perfect_square_condition (a b c k : ℝ) :
  (∃ (f : ℝ → ℝ → ℝ), ∀ (x y : ℝ), a * x^2 + 2 * b * x * y + c * y^2 - k * (x^2 + y^2) = (f x y)^2) ↔
  (k = (a + c) / 2 + (1 / 2) * Real.sqrt ((a - c)^2 + 4 * b^2) ∨
   k = (a + c) / 2 - (1 / 2) * Real.sqrt ((a - c)^2 + 4 * b^2)) :=
by sorry

end perfect_square_condition_l2266_226666


namespace table_height_l2266_226689

/-- Represents the configuration of two identical blocks on a table -/
structure BlockConfiguration where
  l : ℝ  -- length of each block
  w : ℝ  -- width of each block
  h : ℝ  -- height of the table

/-- The length measurement in Figure 1 -/
def figure1_length (config : BlockConfiguration) : ℝ :=
  config.l + config.h - config.w

/-- The length measurement in Figure 2 -/
def figure2_length (config : BlockConfiguration) : ℝ :=
  config.w + config.h - config.l

/-- The main theorem stating the height of the table -/
theorem table_height (config : BlockConfiguration) 
  (h1 : figure1_length config = 32)
  (h2 : figure2_length config = 28) : 
  config.h = 30 := by
  sorry


end table_height_l2266_226689


namespace oil_quantity_function_correct_l2266_226617

/-- Represents the remaining oil quantity in liters -/
def Q (t : ℝ) : ℝ := 40 - 0.2 * t

/-- The initial oil quantity in liters -/
def initial_quantity : ℝ := 40

/-- The oil flow rate in liters per minute -/
def flow_rate : ℝ := 0.2

theorem oil_quantity_function_correct (t : ℝ) :
  Q t = initial_quantity - flow_rate * t :=
by sorry

end oil_quantity_function_correct_l2266_226617


namespace prime_product_sum_proper_fractions_l2266_226692

/-- Sum of proper fractions with denominator k -/
def sum_proper_fractions (k : ℕ) : ℚ :=
  (k - 1) / 2

theorem prime_product_sum_proper_fractions : 
  ∀ m n : ℕ, 
  m.Prime → n.Prime → m < n → 
  (sum_proper_fractions m) * (sum_proper_fractions n) = 5 → 
  m = 3 ∧ n = 11 := by
  sorry

end prime_product_sum_proper_fractions_l2266_226692


namespace smallest_power_l2266_226624

theorem smallest_power (a b c d : ℕ) : 
  a = 2 → b = 3 → c = 5 → d = 6 → 
  a^55 < b^44 ∧ a^55 < c^33 ∧ a^55 < d^22 := by
  sorry

end smallest_power_l2266_226624


namespace unique_k_value_l2266_226686

/-- The polynomial expression -/
def polynomial (k : ℚ) (x y : ℚ) : ℚ := x^2 + 4*x*y + 2*x + k*y - 3*k

/-- Condition for integer factorization -/
def has_integer_factorization (k : ℚ) : Prop :=
  ∃ (A B C D E F : ℤ), 
    ∀ (x y : ℚ), polynomial k x y = (A*x + B*y + C) * (D*x + E*y + F)

/-- Condition for non-negative discriminant of the quadratic part -/
def has_nonnegative_discriminant (k : ℚ) : Prop :=
  (4:ℚ)^2 - 4*1*0 ≥ 0

/-- The main theorem -/
theorem unique_k_value : 
  (∃! k : ℚ, has_integer_factorization k ∧ has_nonnegative_discriminant k) ∧
  (∀ k : ℚ, has_integer_factorization k ∧ has_nonnegative_discriminant k → k = 0) :=
sorry

end unique_k_value_l2266_226686


namespace sin_5pi_6_minus_2alpha_l2266_226678

theorem sin_5pi_6_minus_2alpha (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -1 / 3 := by
  sorry

end sin_5pi_6_minus_2alpha_l2266_226678


namespace point_slope_theorem_l2266_226650

theorem point_slope_theorem (k : ℝ) (h1 : k > 0) : 
  (2 - k) / (k - 1) = k^2 → k = 1 := by sorry

end point_slope_theorem_l2266_226650


namespace fraction_addition_l2266_226629

theorem fraction_addition (d : ℝ) : (3 + 4 * d) / 5 + 3 = (18 + 4 * d) / 5 := by
  sorry

end fraction_addition_l2266_226629


namespace calculator_sum_l2266_226644

/-- The number of participants in the circle. -/
def n : ℕ := 44

/-- The operation performed on the first calculator (squaring). -/
def op1 (x : ℕ) : ℕ := x ^ 2

/-- The operation performed on the second calculator (squaring). -/
def op2 (x : ℕ) : ℕ := x ^ 2

/-- The operation performed on the third calculator (negation). -/
def op3 (x : ℤ) : ℤ := -x

/-- The final value of the first calculator after n iterations. -/
def final1 : ℕ := 2 ^ (2 ^ n)

/-- The final value of the second calculator after n iterations. -/
def final2 : ℕ := 0

/-- The final value of the third calculator after n iterations. -/
def final3 : ℤ := (-1) ^ n

/-- The theorem stating the final sum of the calculators. -/
theorem calculator_sum :
  (final1 : ℤ) + final2 + final3 = 2 ^ (2 ^ n) + 1 := by sorry

end calculator_sum_l2266_226644


namespace smallest_constant_term_l2266_226654

theorem smallest_constant_term (a b c d e : ℤ) : 
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -3 ∨ x = 6 ∨ x = 10 ∨ x = -1/4) →
  e > 0 →
  e ≥ 180 :=
sorry

end smallest_constant_term_l2266_226654


namespace no_lcm_83_l2266_226691

theorem no_lcm_83 (a b c : ℕ) : 
  a = 23 → b = 46 → Nat.lcm a (Nat.lcm b c) = 83 → False :=
by
  sorry

#check no_lcm_83

end no_lcm_83_l2266_226691


namespace min_value_of_function_l2266_226635

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  (x^2 + x + 1) / (x - 1) ≥ 3 + 2 * Real.sqrt 3 := by
  sorry

end min_value_of_function_l2266_226635


namespace usual_time_to_catch_bus_l2266_226675

/-- The usual time to catch the bus, given that walking at 3/5 of the usual speed results in missing the bus by 5 minutes -/
theorem usual_time_to_catch_bus : ∃ (T : ℝ), T > 0 ∧ (5/3 * T = T + 5) ∧ T = 7.5 := by
  sorry

end usual_time_to_catch_bus_l2266_226675


namespace count_quadrilaterals_with_equidistant_point_l2266_226690

/-- A quadrilateral in a plane -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A point is equidistant from all vertices of a quadrilateral -/
def has_equidistant_point (q : Quadrilateral) : Prop :=
  ∃ p : ℝ × ℝ, ∀ i : Fin 4, dist p (q.vertices i) = dist p (q.vertices 0)

/-- A kite with two consecutive right angles -/
def is_kite_with_two_right_angles (q : Quadrilateral) : Prop :=
  sorry

/-- A rectangle with sides in the ratio 3:1 -/
def is_rectangle_3_1 (q : Quadrilateral) : Prop :=
  sorry

/-- A rhombus with an angle of 120 degrees -/
def is_rhombus_120 (q : Quadrilateral) : Prop :=
  sorry

/-- A general quadrilateral with perpendicular diagonals -/
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  sorry

/-- An isosceles trapezoid where the non-parallel sides are equal in length -/
def is_isosceles_trapezoid (q : Quadrilateral) : Prop :=
  sorry

/-- The main theorem -/
theorem count_quadrilaterals_with_equidistant_point :
  ∃ (q1 q2 q3 : Quadrilateral),
    (is_kite_with_two_right_angles q1 ∨ 
     is_rectangle_3_1 q1 ∨ 
     is_rhombus_120 q1 ∨ 
     has_perpendicular_diagonals q1 ∨ 
     is_isosceles_trapezoid q1) ∧
    (is_kite_with_two_right_angles q2 ∨ 
     is_rectangle_3_1 q2 ∨ 
     is_rhombus_120 q2 ∨ 
     has_perpendicular_diagonals q2 ∨ 
     is_isosceles_trapezoid q2) ∧
    (is_kite_with_two_right_angles q3 ∨ 
     is_rectangle_3_1 q3 ∨ 
     is_rhombus_120 q3 ∨ 
     has_perpendicular_diagonals q3 ∨ 
     is_isosceles_trapezoid q3) ∧
    has_equidistant_point q1 ∧
    has_equidistant_point q2 ∧
    has_equidistant_point q3 ∧
    (∀ q : Quadrilateral, 
      (is_kite_with_two_right_angles q ∨ 
       is_rectangle_3_1 q ∨ 
       is_rhombus_120 q ∨ 
       has_perpendicular_diagonals q ∨ 
       is_isosceles_trapezoid q) →
      has_equidistant_point q →
      (q = q1 ∨ q = q2 ∨ q = q3)) :=
by
  sorry

end count_quadrilaterals_with_equidistant_point_l2266_226690


namespace montague_population_fraction_l2266_226670

/-- The fraction of the population living in Montague province -/
def montague_fraction : ℝ := sorry

/-- The fraction of the population living in Capulet province -/
def capulet_fraction : ℝ := sorry

/-- The theorem stating the conditions and the result to be proved -/
theorem montague_population_fraction :
  -- Conditions
  (montague_fraction + capulet_fraction = 1) ∧
  (0.8 * montague_fraction + 0.3 * capulet_fraction = 0.7 * capulet_fraction / (7/11)) →
  -- Conclusion
  montague_fraction = 2/3 := by sorry

end montague_population_fraction_l2266_226670


namespace total_team_combinations_l2266_226656

/-- The number of ways to choose k items from n items without replacement and without regard to order. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of people in each group. -/
def group_size : ℕ := 6

/-- The number of people to be selected from each group. -/
def team_size : ℕ := 3

/-- The number of groups. -/
def num_groups : ℕ := 2

theorem total_team_combinations : 
  (choose group_size team_size) ^ num_groups = 400 := by sorry

end total_team_combinations_l2266_226656


namespace fruit_cost_prices_l2266_226613

/-- Represents the cost and selling prices of fruits -/
structure FruitPrices where
  apple_sell : ℚ
  orange_sell : ℚ
  banana_sell : ℚ
  apple_loss : ℚ
  orange_loss : ℚ
  banana_gain : ℚ

/-- Calculates the cost price given selling price and loss/gain percentage -/
def cost_price (sell : ℚ) (loss_gain : ℚ) (is_gain : Bool) : ℚ :=
  if is_gain then
    sell / (1 + loss_gain)
  else
    sell / (1 - loss_gain)

/-- Theorem stating the correct cost prices for the fruits -/
theorem fruit_cost_prices (prices : FruitPrices)
  (h_apple_sell : prices.apple_sell = 20)
  (h_orange_sell : prices.orange_sell = 15)
  (h_banana_sell : prices.banana_sell = 6)
  (h_apple_loss : prices.apple_loss = 1/6)
  (h_orange_loss : prices.orange_loss = 1/4)
  (h_banana_gain : prices.banana_gain = 1/8) :
  cost_price prices.apple_sell prices.apple_loss false = 24 ∧
  cost_price prices.orange_sell prices.orange_loss false = 20 ∧
  cost_price prices.banana_sell prices.banana_gain true = 16/3 := by
  sorry

end fruit_cost_prices_l2266_226613


namespace jack_driving_distance_l2266_226634

/-- Calculates the number of miles driven every four months given the total years of driving and total miles driven. -/
def miles_per_four_months (years : ℕ) (total_miles : ℕ) : ℕ :=
  total_miles / (years * 3)

/-- Theorem stating that driving for 9 years and covering 999,000 miles results in driving 37,000 miles every four months. -/
theorem jack_driving_distance :
  miles_per_four_months 9 999000 = 37000 := by
  sorry

end jack_driving_distance_l2266_226634


namespace log_equality_l2266_226622

theorem log_equality (x : ℝ) (h : x > 0) :
  (Real.log (2 * x) / Real.log (5 * x) = Real.log (8 * x) / Real.log (625 * x)) →
  (Real.log x / Real.log 2 = Real.log 5 / (3 * (Real.log 2 - Real.log 5))) := by
  sorry

end log_equality_l2266_226622


namespace odd_terms_in_binomial_expansion_l2266_226693

theorem odd_terms_in_binomial_expansion (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  (Finset.range 9).filter (fun k => Odd (Nat.choose 8 k * (p + q)^(8 - k) * p^k)) = {0, 8} := by
  sorry

end odd_terms_in_binomial_expansion_l2266_226693


namespace car_speed_problem_l2266_226633

/-- Given a car traveling for two hours, where its speed in the second hour is 30 km/h
    and its average speed over the two hours is 25 km/h, prove that the speed of the car
    in the first hour must be 20 km/h. -/
theorem car_speed_problem (first_hour_speed : ℝ) : 
  (first_hour_speed + 30) / 2 = 25 → first_hour_speed = 20 := by
  sorry

end car_speed_problem_l2266_226633


namespace min_value_sum_reciprocals_no_solution_product_equation_l2266_226673

/-- Given x and y are positive real numbers satisfying x^2 + y^2 = x + y -/
def satisfies_equation (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x^2 + y^2 = x + y

/-- The minimum value of 1/x + 1/y is 2 -/
theorem min_value_sum_reciprocals {x y : ℝ} (h : satisfies_equation x y) :
  1/x + 1/y ≥ 2 := by
  sorry

/-- There do not exist x and y satisfying (x+1)(y+1) = 5 -/
theorem no_solution_product_equation {x y : ℝ} (h : satisfies_equation x y) :
  (x + 1) * (y + 1) ≠ 5 := by
  sorry

end min_value_sum_reciprocals_no_solution_product_equation_l2266_226673


namespace book_arrangement_theorem_l2266_226658

def total_books : ℕ := 7
def science_books : ℕ := 2
def math_books : ℕ := 2
def unique_books : ℕ := total_books - science_books - math_books

def arrangements : ℕ := (total_books.factorial) / (science_books.factorial * math_books.factorial)

def highlighted_arrangements : ℕ := arrangements * (total_books.choose 2)

theorem book_arrangement_theorem :
  arrangements = 1260 ∧ highlighted_arrangements = 26460 := by
  sorry

end book_arrangement_theorem_l2266_226658


namespace max_product_under_constraint_l2266_226611

theorem max_product_under_constraint (a b : ℝ) :
  a > 0 → b > 0 → 5 * a + 8 * b = 80 → ab ≤ 40 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ 5 * a + 8 * b = 80 ∧ a * b = 40 := by
  sorry

end max_product_under_constraint_l2266_226611


namespace sugar_amount_proof_l2266_226646

/-- The amount of sugar in pounds in the first combination -/
def sugar_amount : ℝ := 39

/-- The cost per pound of sugar and flour in dollars -/
def cost_per_pound : ℝ := 0.45

/-- The cost of the first combination in dollars -/
def cost_first : ℝ := 26

/-- The cost of the second combination in dollars -/
def cost_second : ℝ := 26

/-- The amount of flour in the first combination in pounds -/
def flour_first : ℝ := 16

/-- The amount of sugar in the second combination in pounds -/
def sugar_second : ℝ := 30

/-- The amount of flour in the second combination in pounds -/
def flour_second : ℝ := 25

theorem sugar_amount_proof :
  cost_per_pound * sugar_amount + cost_per_pound * flour_first = cost_first ∧
  cost_per_pound * sugar_second + cost_per_pound * flour_second = cost_second ∧
  sugar_amount + flour_first = sugar_second + flour_second :=
by sorry

end sugar_amount_proof_l2266_226646


namespace vector_sum_magnitude_l2266_226679

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

theorem vector_sum_magnitude (x : ℝ) 
  (h : vector_a • vector_b x = -3) : 
  ‖vector_a + vector_b x‖ = 2 := by
sorry

end vector_sum_magnitude_l2266_226679


namespace exam_grading_rules_l2266_226603

-- Define the types
def Student : Type := String
def Grade : Type := String
def Essay : Type := Bool

-- Define the predicates
def all_mc_correct (s : Student) : Prop := sorry
def satisfactory_essay (s : Student) : Prop := sorry
def grade_is (s : Student) (g : Grade) : Prop := sorry

-- State the theorem
theorem exam_grading_rules (s : Student) :
  -- Condition 1
  (∀ s, all_mc_correct s → grade_is s "B" ∨ grade_is s "A") →
  -- Condition 2
  (∀ s, all_mc_correct s ∧ satisfactory_essay s → grade_is s "A") →
  -- Statement D
  (grade_is s "A" → all_mc_correct s) ∧
  -- Statement E
  (all_mc_correct s ∧ satisfactory_essay s → grade_is s "A") := by
  sorry


end exam_grading_rules_l2266_226603


namespace power_multiplication_l2266_226604

theorem power_multiplication (a b : ℕ) : (10 : ℕ) ^ 85 * (10 : ℕ) ^ 84 = (10 : ℕ) ^ (85 + 84) := by
  sorry

end power_multiplication_l2266_226604


namespace relay_race_distance_l2266_226631

theorem relay_race_distance (total_distance : ℕ) (team_members : ℕ) (individual_distance : ℕ) : 
  total_distance = 150 ∧ team_members = 5 ∧ individual_distance * team_members = total_distance →
  individual_distance = 30 := by
  sorry

end relay_race_distance_l2266_226631


namespace binomial_nine_choose_five_l2266_226651

theorem binomial_nine_choose_five : Nat.choose 9 5 = 126 := by sorry

end binomial_nine_choose_five_l2266_226651


namespace max_value_implies_b_equals_two_l2266_226630

def is_valid_triple (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a ∈ ({2, 3, 6} : Set ℕ) ∧ 
  b ∈ ({2, 3, 6} : Set ℕ) ∧ 
  c ∈ ({2, 3, 6} : Set ℕ)

theorem max_value_implies_b_equals_two (a b c : ℕ) :
  is_valid_triple a b c →
  (a : ℚ) / (b / c) ≤ 9 →
  (∀ x y z : ℕ, is_valid_triple x y z → (x : ℚ) / (y / z) ≤ 9) →
  b = 2 :=
sorry

end max_value_implies_b_equals_two_l2266_226630


namespace nba_division_impossibility_l2266_226643

theorem nba_division_impossibility : ∀ (A B : ℕ),
  A + B = 30 →
  A * B ≠ (30 * 82) / 4 :=
by
  sorry


end nba_division_impossibility_l2266_226643


namespace ice_pop_probability_l2266_226648

/-- Represents the number of ice pops of each flavor --/
structure IcePops where
  cherry : ℕ
  orange : ℕ
  lemonLime : ℕ

/-- Calculates the probability of selecting two ice pops of different flavors --/
def probDifferentFlavors (pops : IcePops) : ℚ :=
  let total := pops.cherry + pops.orange + pops.lemonLime
  1 - (pops.cherry * (pops.cherry - 1) + pops.orange * (pops.orange - 1) + pops.lemonLime * (pops.lemonLime - 1)) / (total * (total - 1))

/-- The main theorem stating that for the given ice pop distribution, 
    the probability of selecting two different flavors is 8/11 --/
theorem ice_pop_probability : 
  let pops : IcePops := ⟨4, 3, 4⟩
  probDifferentFlavors pops = 8/11 := by
  sorry

end ice_pop_probability_l2266_226648


namespace leading_coefficient_of_P_l2266_226667

/-- The polynomial in question -/
def P (x : ℝ) : ℝ := -5 * (x^4 - 2*x^3 + 3*x) + 8 * (x^4 - x^2 + 1) - 3 * (3*x^4 + x^3 + x)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : ℝ → ℝ) : ℝ :=
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_P :
  leadingCoefficient P = -6 := by
  sorry

end leading_coefficient_of_P_l2266_226667


namespace gcd_of_polynomial_and_multiple_of_570_l2266_226600

theorem gcd_of_polynomial_and_multiple_of_570 (b : ℤ) : 
  (∃ k : ℤ, b = 570 * k) → Int.gcd (4 * b^3 + 2 * b^2 + 5 * b + 95) b = 95 := by
  sorry

end gcd_of_polynomial_and_multiple_of_570_l2266_226600


namespace martha_troubleshooting_time_l2266_226663

/-- The total time Martha spent on router troubleshooting activities -/
def total_time (router_time hold_time yelling_time : ℕ) : ℕ :=
  router_time + hold_time + yelling_time

/-- Theorem stating the total time Martha spent on activities -/
theorem martha_troubleshooting_time :
  ∃ (router_time hold_time yelling_time : ℕ),
    router_time = 10 ∧
    hold_time = 6 * router_time ∧
    yelling_time = hold_time / 2 ∧
    total_time router_time hold_time yelling_time = 100 := by
  sorry

end martha_troubleshooting_time_l2266_226663


namespace union_of_A_and_B_l2266_226681

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = Set.Icc (-1) 2 := by sorry

end union_of_A_and_B_l2266_226681


namespace largest_m_l2266_226645

/-- A three-digit positive integer that is the product of three distinct prime factors --/
def m (x y : ℕ) : ℕ := x * y * (10 * x + y)

/-- The proposition that m is the largest possible value given the conditions --/
theorem largest_m : ∀ x y : ℕ, 
  x < 10 → y < 10 → x ≠ y → 
  Nat.Prime x → Nat.Prime y → Nat.Prime (10 * x + y) →
  m x y ≤ 795 ∧ m x y < 1000 :=
sorry

end largest_m_l2266_226645


namespace sandy_fingernail_record_l2266_226632

/-- Calculates the length of fingernails after a given number of years -/
def fingernail_length (current_age : ℕ) (target_age : ℕ) (current_length : ℝ) (growth_rate : ℝ) : ℝ :=
  current_length + (target_age - current_age) * 12 * growth_rate

/-- Proves that Sandy's fingernails will be 26 inches long at age 32, given the initial conditions -/
theorem sandy_fingernail_record :
  fingernail_length 12 32 2 0.1 = 26 := by
  sorry

end sandy_fingernail_record_l2266_226632


namespace floor_sqrt_50_squared_l2266_226698

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end floor_sqrt_50_squared_l2266_226698


namespace shop_pricing_l2266_226638

theorem shop_pricing (CP : ℝ) 
  (h1 : CP * 0.5 = 320) : CP * 1.25 = 800 := by
  sorry

end shop_pricing_l2266_226638


namespace base_eight_perfect_square_c_is_one_l2266_226640

/-- Represents a number in base 8 with the form 1b27c -/
def BaseEightNumber (b c : ℕ) : ℕ := 1024 + 64 * b + 16 + 7 + c

/-- A number is a perfect square if there exists an integer whose square is that number -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- The last digit of a perfect square in base 8 can only be 0, 1, or 4 -/
axiom perfect_square_mod_8 (n : ℕ) : IsPerfectSquare n → n % 8 ∈ ({0, 1, 4} : Set ℕ)

theorem base_eight_perfect_square_c_is_one (b : ℕ) :
  IsPerfectSquare (BaseEightNumber b 1) →
  ∀ c : ℕ, IsPerfectSquare (BaseEightNumber b c) → c = 1 := by
  sorry

end base_eight_perfect_square_c_is_one_l2266_226640


namespace amount_with_r_l2266_226641

theorem amount_with_r (total : ℝ) (p q r : ℝ) : 
  total = 9000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 3600 := by
sorry

end amount_with_r_l2266_226641


namespace quadratic_expression_value_l2266_226642

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 8) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 100 := by
  sorry

end quadratic_expression_value_l2266_226642


namespace constant_equation_solution_l2266_226607

theorem constant_equation_solution (n : ℝ) : 
  (∃ m : ℝ, 21 * (m + n) + 21 = 21 * (-m + n) + 21) → 
  (∃ m : ℝ, m = 0 ∧ 21 * (m + n) + 21 = 21 * (-m + n) + 21) :=
by sorry

end constant_equation_solution_l2266_226607


namespace apples_count_l2266_226684

/-- The number of apples in the market -/
def apples : ℕ := sorry

/-- The number of oranges in the market -/
def oranges : ℕ := sorry

/-- The number of bananas in the market -/
def bananas : ℕ := sorry

/-- There are 27 more apples than oranges -/
axiom apples_oranges_diff : apples = oranges + 27

/-- There are 11 more oranges than bananas -/
axiom oranges_bananas_diff : oranges = bananas + 11

/-- The total number of fruits is 301 -/
axiom total_fruits : apples + oranges + bananas = 301

/-- The number of apples in the market is 122 -/
theorem apples_count : apples = 122 := by sorry

end apples_count_l2266_226684


namespace area_at_stage_5_is_24_l2266_226661

/-- Calculates the length of the rectangle at a given stage -/
def length_at_stage (stage : ℕ) : ℕ := 4 + 2 * (stage - 1)

/-- Calculates the area of the rectangle at a given stage -/
def area_at_stage (stage : ℕ) : ℕ := length_at_stage stage * 2

/-- Theorem stating that the area at Stage 5 is 24 square inches -/
theorem area_at_stage_5_is_24 : area_at_stage 5 = 24 := by
  sorry

#eval area_at_stage 5  -- This should output 24

end area_at_stage_5_is_24_l2266_226661


namespace quadratic_inequality_l2266_226647

theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by sorry

end quadratic_inequality_l2266_226647


namespace imaginary_unit_power_2018_l2266_226620

theorem imaginary_unit_power_2018 (i : ℂ) (hi : i^2 = -1) : i^2018 = -1 := by
  sorry

end imaginary_unit_power_2018_l2266_226620


namespace doughnuts_given_away_l2266_226636

def doughnuts_per_box : ℕ := 10
def total_doughnuts : ℕ := 300
def boxes_sold : ℕ := 27

theorem doughnuts_given_away : ℕ := by
  have h1 : total_doughnuts % doughnuts_per_box = 0 := by sorry
  have h2 : total_doughnuts / doughnuts_per_box > boxes_sold := by sorry
  exact (total_doughnuts / doughnuts_per_box - boxes_sold) * doughnuts_per_box

end doughnuts_given_away_l2266_226636


namespace unique_valid_number_l2266_226694

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    b = 3 ∧
    a + c = 5 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∃ (d : ℕ), n + 124 = 111 * d

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n ∧ n = 431 :=
sorry

end unique_valid_number_l2266_226694


namespace negation_of_square_nonnegative_l2266_226682

theorem negation_of_square_nonnegative :
  (¬ ∀ x : ℝ, x ^ 2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀ ^ 2 < 0) := by sorry

end negation_of_square_nonnegative_l2266_226682


namespace double_age_in_two_years_l2266_226653

/-- Represents the number of years until a man's age is twice his son's age. -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  let x := man_age + 2 - 2 * (son_age + 2)
  2

/-- Theorem stating that the number of years until the man's age is twice his son's age is 2,
    given the son's current age and the age difference between the man and his son. -/
theorem double_age_in_two_years (son_age : ℕ) (age_difference : ℕ) 
    (h1 : son_age = 28) (h2 : age_difference = 30) : 
    years_until_double_age son_age age_difference = 2 := by
  sorry

#eval years_until_double_age 28 30

end double_age_in_two_years_l2266_226653


namespace different_color_probability_l2266_226639

def total_chips : ℕ := 15
def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

theorem different_color_probability : 
  let prob_blue_not_blue := (blue_chips : ℚ) / total_chips * ((red_chips + yellow_chips) : ℚ) / total_chips
  let prob_red_not_red := (red_chips : ℚ) / total_chips * ((blue_chips + yellow_chips) : ℚ) / total_chips
  let prob_yellow_not_yellow := (yellow_chips : ℚ) / total_chips * ((blue_chips + red_chips) : ℚ) / total_chips
  prob_blue_not_blue + prob_red_not_red + prob_yellow_not_yellow = 148 / 225 :=
by sorry

end different_color_probability_l2266_226639


namespace fraction_equation_solution_l2266_226610

theorem fraction_equation_solution (x : ℚ) :
  2/5 - 1/4 = 1/x → x = 20/3 := by
  sorry

end fraction_equation_solution_l2266_226610


namespace greenhouse_renovation_l2266_226649

/-- Greenhouse renovation problem -/
theorem greenhouse_renovation 
  (cost_2A_vs_1B : ℝ) 
  (cost_1A_2B : ℝ) 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (total_greenhouses : ℕ) 
  (max_budget : ℝ) 
  (max_days : ℝ)
  (h1 : cost_2A_vs_1B = 6)
  (h2 : cost_1A_2B = 48)
  (h3 : days_A = 5)
  (h4 : days_B = 3)
  (h5 : total_greenhouses = 8)
  (h6 : max_budget = 128)
  (h7 : max_days = 35) :
  ∃ (cost_A cost_B : ℝ),
    cost_A = 12 ∧ 
    cost_B = 18 ∧
    2 * cost_A = cost_B + cost_2A_vs_1B ∧
    cost_A + 2 * cost_B = cost_1A_2B ∧
    (∀ m : ℕ, 
      (m ≤ total_greenhouses ∧
       m * cost_A + (total_greenhouses - m) * cost_B ≤ max_budget ∧
       m * days_A + (total_greenhouses - m) * days_B ≤ max_days) 
      ↔ m ∈ ({3, 4, 5} : Set ℕ)) :=
sorry

end greenhouse_renovation_l2266_226649


namespace solve_for_b_l2266_226619

theorem solve_for_b (a b : ℚ) (h1 : a = 5) (h2 : b - a + (2 * b / 3) = 7) : b = 36 / 5 := by
  sorry

end solve_for_b_l2266_226619


namespace rectangular_field_area_l2266_226655

/-- A rectangular field with length double its width and perimeter 180 meters has an area of 1800 square meters. -/
theorem rectangular_field_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  width > 0 →
  length = 2 * width →
  perimeter = 2 * (length + width) →
  perimeter = 180 →
  area = length * width →
  area = 1800 := by
  sorry

#check rectangular_field_area

end rectangular_field_area_l2266_226655


namespace abs_inequality_iff_inequality_l2266_226606

theorem abs_inequality_iff_inequality (a b : ℝ) : a > b ↔ a * |a| > b * |b| := by sorry

end abs_inequality_iff_inequality_l2266_226606


namespace judy_school_week_days_l2266_226627

/-- The number of pencils Judy uses during her school week. -/
def pencils_per_week : ℕ := 10

/-- The number of pencils in a pack. -/
def pencils_per_pack : ℕ := 30

/-- The cost of a pack of pencils in dollars. -/
def cost_per_pack : ℚ := 4

/-- The amount Judy spends on pencils in dollars. -/
def total_spent : ℚ := 12

/-- The number of days over which Judy spends the total amount. -/
def total_days : ℕ := 45

/-- The number of days in Judy's school week. -/
def school_week_days : ℕ := 5

/-- Theorem stating that the number of days in Judy's school week is 5. -/
theorem judy_school_week_days :
  (pencils_per_week : ℚ) * total_days * cost_per_pack =
  pencils_per_pack * total_spent * (school_week_days : ℚ) :=
by sorry

end judy_school_week_days_l2266_226627


namespace constant_vertex_l2266_226685

/-- The function f(x) = a^(x-2) + 1 always passes through the point (2, 2) for a > 0 and a ≠ 1 -/
theorem constant_vertex (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 1
  f 2 = 2 := by
  sorry

end constant_vertex_l2266_226685


namespace unique_number_with_conditions_l2266_226609

theorem unique_number_with_conditions : ∃! N : ℤ,
  35 < N ∧ N < 70 ∧ N % 6 = 3 ∧ N % 8 = 1 ∧ N = 57 := by
  sorry

end unique_number_with_conditions_l2266_226609


namespace product_of_decimals_l2266_226657

theorem product_of_decimals (h : 268 * 74 = 19832) :
  2.68 * 0.74 = 1.9832 := by
  sorry

end product_of_decimals_l2266_226657


namespace base_b_square_l2266_226674

theorem base_b_square (b : ℕ) (h : b > 1) : 
  (3 * b^2 + 4 * b + 3 = (b + 3)^2) → b = 3 := by
  sorry

end base_b_square_l2266_226674


namespace smurfs_gold_coins_l2266_226608

theorem smurfs_gold_coins (total : ℕ) (smurfs : ℕ) (gargamel : ℕ) 
  (h1 : total = 200)
  (h2 : smurfs + gargamel = total)
  (h3 : (2 : ℚ) / 3 * smurfs = (4 : ℚ) / 5 * gargamel + 38) :
  smurfs = 135 := by
  sorry

end smurfs_gold_coins_l2266_226608


namespace expression_evaluation_l2266_226652

theorem expression_evaluation : 
  Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3) + 
  Real.sqrt (7 + 2 * Real.sqrt 10) - Real.sqrt (7 - 2 * Real.sqrt 10) = 2 * Real.sqrt 2 := by
  sorry

end expression_evaluation_l2266_226652


namespace real_square_nonnegative_and_no_real_square_root_of_negative_one_l2266_226669

theorem real_square_nonnegative_and_no_real_square_root_of_negative_one :
  (∀ x : ℝ, x^2 ≥ 0) ∧ ¬(∃ x : ℝ, x^2 = -1) := by
  sorry

end real_square_nonnegative_and_no_real_square_root_of_negative_one_l2266_226669


namespace least_integer_square_52_more_than_triple_l2266_226616

theorem least_integer_square_52_more_than_triple : 
  ∃ x : ℤ, x^2 = 3*x + 52 ∧ ∀ y : ℤ, y^2 = 3*y + 52 → x ≤ y :=
by sorry

end least_integer_square_52_more_than_triple_l2266_226616


namespace road_trip_distance_ratio_l2266_226601

theorem road_trip_distance_ratio : 
  ∀ (total_distance first_day_distance second_day_distance third_day_distance : ℝ),
  total_distance = 525 →
  first_day_distance = 200 →
  second_day_distance = 3/4 * first_day_distance →
  third_day_distance = total_distance - (first_day_distance + second_day_distance) →
  third_day_distance / (first_day_distance + second_day_distance) = 1/2 := by
sorry

end road_trip_distance_ratio_l2266_226601


namespace inequality_proof_l2266_226687

theorem inequality_proof (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos₁ : a₁ > 0) (h_pos₂ : a₂ > 0) (h_pos₃ : a₃ > 0) (h_pos₄ : a₄ > 0)
  (h_distinct₁₂ : a₁ ≠ a₂) (h_distinct₁₃ : a₁ ≠ a₃) (h_distinct₁₄ : a₁ ≠ a₄)
  (h_distinct₂₃ : a₂ ≠ a₃) (h_distinct₂₄ : a₂ ≠ a₄) (h_distinct₃₄ : a₃ ≠ a₄) :
  a₁^3 / (a₂ - a₃)^2 + a₂^3 / (a₃ - a₄)^2 + a₃^3 / (a₄ - a₁)^2 + a₄^3 / (a₁ - a₂)^2 
  > a₁ + a₂ + a₃ + a₄ :=
by sorry

end inequality_proof_l2266_226687


namespace max_value_abc_l2266_226668

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  a^4 * b^3 * c^2 ≤ 1 / 6561 := by
sorry

end max_value_abc_l2266_226668


namespace maxwell_walking_speed_l2266_226605

/-- Prove that Maxwell's walking speed is 4 km/h given the conditions of the problem -/
theorem maxwell_walking_speed :
  ∀ (maxwell_speed : ℝ),
    maxwell_speed > 0 →
    (3 * maxwell_speed) + (2 * 6) = 24 →
    maxwell_speed = 4 :=
by
  sorry

#check maxwell_walking_speed

end maxwell_walking_speed_l2266_226605


namespace ratio_of_bases_l2266_226664

/-- An isosceles trapezoid circumscribed about a circle -/
structure CircumscribedTrapezoid where
  /-- The longer base of the trapezoid -/
  AD : ℝ
  /-- The shorter base of the trapezoid -/
  BC : ℝ
  /-- The ratio of AN to NM, where N is the intersection of AM and the circle -/
  k : ℝ
  /-- AD is longer than BC -/
  h_AD_gt_BC : AD > BC
  /-- The trapezoid is isosceles -/
  h_isosceles : True
  /-- The trapezoid is circumscribed about a circle -/
  h_circumscribed : True
  /-- The circle touches one of the non-parallel sides -/
  h_touches_side : True
  /-- AM intersects the circle at N -/
  h_AM_intersects : True

/-- The ratio of the longer base to the shorter base in a circumscribed isosceles trapezoid -/
theorem ratio_of_bases (t : CircumscribedTrapezoid) : t.AD / t.BC = 8 * t.k - 1 := by
  sorry

end ratio_of_bases_l2266_226664


namespace square_division_negative_numbers_l2266_226662

theorem square_division_negative_numbers : (-128)^2 / (-64)^2 = 4 := by
  sorry

end square_division_negative_numbers_l2266_226662


namespace smaller_rectangle_area_l2266_226618

/-- The area of a rectangle with half the length and half the width of a 40m by 20m rectangle is 200 square meters. -/
theorem smaller_rectangle_area (big_length big_width : ℝ) 
  (h_big_length : big_length = 40)
  (h_big_width : big_width = 20)
  (small_length small_width : ℝ)
  (h_small_length : small_length = big_length / 2)
  (h_small_width : small_width = big_width / 2) :
  small_length * small_width = 200 := by
  sorry

end smaller_rectangle_area_l2266_226618


namespace geometric_sequence_third_term_l2266_226699

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_first : a 1 = 1)
  (h_fifth : a 5 = 5) :
  a 3 = Real.sqrt 5 :=
sorry

end geometric_sequence_third_term_l2266_226699


namespace socks_cost_theorem_l2266_226602

/-- The cost of each red pair of socks -/
def red_cost : ℝ := 3

/-- The number of red sock pairs -/
def red_pairs : ℕ := 4

/-- The number of blue sock pairs -/
def blue_pairs : ℕ := 6

/-- The cost of each blue pair of socks -/
def blue_cost : ℝ := 5

/-- The total cost of all socks -/
def total_cost : ℝ := 42

theorem socks_cost_theorem :
  red_cost * red_pairs + blue_cost * blue_pairs = total_cost :=
by sorry

end socks_cost_theorem_l2266_226602


namespace arithmetic_sequence_properties_l2266_226696

/-- An arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

theorem arithmetic_sequence_properties (a d : ℝ) :
  let seq := arithmetic_sequence a d
  (∀ n : ℕ, n > 0 → seq (n + 1) - seq n = d) ∧
  (seq 4 = 15 ∧ seq 15 = 59) := by sorry

end arithmetic_sequence_properties_l2266_226696


namespace andrews_sleepover_donuts_l2266_226637

/-- The number of donuts Andrew's mother needs to buy for a sleepover --/
def donuts_for_sleepover (initial_friends : ℕ) (additional_friends : ℕ) (donuts_per_friend : ℕ) (extra_donuts_per_friend : ℕ) : ℕ :=
  let total_friends := initial_friends + additional_friends
  let donuts_for_friends := total_friends * (donuts_per_friend + extra_donuts_per_friend)
  let donuts_for_andrew := donuts_per_friend + extra_donuts_per_friend
  donuts_for_friends + donuts_for_andrew

/-- Theorem: Andrew's mother needs to buy 20 donuts for the sleepover --/
theorem andrews_sleepover_donuts :
  donuts_for_sleepover 2 2 3 1 = 20 := by
  sorry

end andrews_sleepover_donuts_l2266_226637


namespace roll_distribution_probability_l2266_226660

/-- The number of guests -/
def num_guests : ℕ := 4

/-- The number of roll types -/
def num_roll_types : ℕ := 4

/-- The total number of rolls -/
def total_rolls : ℕ := 16

/-- The number of rolls per type -/
def rolls_per_type : ℕ := 4

/-- The number of rolls given to each guest -/
def rolls_per_guest : ℕ := 4

/-- The probability of each guest receiving one of each type of roll -/
def probability_each_guest_gets_one_of_each : ℚ := 1 / 6028032000

theorem roll_distribution_probability :
  probability_each_guest_gets_one_of_each = 
    (rolls_per_type / total_rolls) *
    ((rolls_per_type - 1) / (total_rolls - 1)) *
    ((rolls_per_type - 2) / (total_rolls - 2)) *
    ((rolls_per_type - 3) / (total_rolls - 3)) *
    ((rolls_per_type - 1) / (total_rolls - 4)) *
    ((rolls_per_type - 2) / (total_rolls - 5)) *
    ((rolls_per_type - 3) / (total_rolls - 6)) *
    ((rolls_per_type - 2) / (total_rolls - 8)) *
    ((rolls_per_type - 3) / (total_rolls - 9)) *
    ((rolls_per_type - 3) / (total_rolls - 12)) := by
  sorry

#eval probability_each_guest_gets_one_of_each

end roll_distribution_probability_l2266_226660


namespace function_properties_l2266_226677

def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def isOddOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f (-x) = -f x

def isLinearOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ m k, ∀ x, a ≤ x ∧ x ≤ b → f x = m * x + k

def isQuadraticOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ a₀ a₁ a₂, ∀ x, a ≤ x ∧ x ≤ b → f x = a₂ * x^2 + a₁ * x + a₀

theorem function_properties (f : ℝ → ℝ) 
    (h1 : isPeriodic f 5)
    (h2 : isOddOn f (-1) 1)
    (h3 : isLinearOn f 0 1)
    (h4 : isQuadraticOn f 1 4)
    (h5 : ∃ x, x = 2 ∧ f x = -5 ∧ ∀ y, f y ≥ -5) :
  (f 1 + f 4 = 0) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 4 → f x = 5/3 * (x - 2)^2 - 5) ∧
  (∀ x, 4 ≤ x ∧ x ≤ 6 → f x = -10/3 * x + 50/3) ∧
  (∀ x, 6 < x ∧ x ≤ 9 → f x = 5/3 * (x - 7)^2 - 5) := by
  sorry

end function_properties_l2266_226677


namespace rationalize_denominator_l2266_226625

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (5 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = A * Real.sqrt B + C * Real.sqrt D ∧
    B < D ∧
    A = -4 ∧
    B = 7 ∧
    C = 3 ∧
    D = 13 ∧
    E = 1 :=
by sorry

end rationalize_denominator_l2266_226625


namespace spotlight_illumination_theorem_l2266_226628

/-- Represents a point on a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction (North, South, East, West) --/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a spotlight that illuminates a right angle --/
structure Spotlight where
  position : Point
  direction1 : Direction
  direction2 : Direction

/-- Represents the configuration of four spotlights --/
structure SpotlightConfiguration where
  spotlights : Fin 4 → Spotlight

/-- Predicate to check if a configuration illuminates the entire plane --/
def illuminatesEntirePlane (config : SpotlightConfiguration) : Prop := sorry

/-- The main theorem stating that there exists a configuration of spotlights that illuminates the entire plane --/
theorem spotlight_illumination_theorem :
  ∃ (config : SpotlightConfiguration), illuminatesEntirePlane config :=
sorry

end spotlight_illumination_theorem_l2266_226628


namespace derivative_of_f_tangent_line_at_one_l2266_226621

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x * Real.log x

-- State the theorem for the derivative of f
theorem derivative_of_f :
  deriv f = fun x => 2 * x + Real.log x + 1 :=
sorry

-- Define the tangent line function
def tangent_line (x y : ℝ) : ℝ := 3 * x - y - 2

-- State the theorem for the tangent line at x=1
theorem tangent_line_at_one :
  ∀ x y, f 1 = y → deriv f 1 * (x - 1) + y = tangent_line x y :=
sorry

end derivative_of_f_tangent_line_at_one_l2266_226621


namespace lobster_theorem_l2266_226676

/-- The total pounds of lobster in three harbors -/
def total_lobster (hooper_bay other1 other2 : ℕ) : ℕ := hooper_bay + other1 + other2

/-- Theorem stating the total pounds of lobster in the three harbors -/
theorem lobster_theorem (hooper_bay other1 other2 : ℕ) 
  (h1 : hooper_bay = 2 * (other1 + other2)) 
  (h2 : other1 = 80) 
  (h3 : other2 = 80) : 
  total_lobster hooper_bay other1 other2 = 480 := by
  sorry

#check lobster_theorem

end lobster_theorem_l2266_226676


namespace a_ge_one_l2266_226626

open Real

/-- The function f(x) = a * ln(x) + (1/2) * x^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + (1/2) * x^2

/-- Theorem stating that if f satisfies the given condition, then a ≥ 1 -/
theorem a_ge_one (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 2) →
  a ≥ 1 :=
by sorry

end a_ge_one_l2266_226626
