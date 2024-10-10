import Mathlib

namespace horizontal_arrangement_possible_l2912_291297

/-- Represents a domino on the board -/
structure Domino where
  x : ℕ
  y : ℕ
  horizontal : Bool

/-- Represents the chessboard with an extra cell -/
structure Board where
  cells : ℕ
  dominoes : List Domino

/-- Checks if a given board configuration is valid -/
def is_valid_board (b : Board) : Prop :=
  b.cells = 65 ∧ b.dominoes.length = 32

/-- Checks if all dominoes on the board are horizontal -/
def all_horizontal (b : Board) : Prop :=
  b.dominoes.all (λ d => d.horizontal)

/-- Represents the ability to move dominoes on the board -/
def can_move_domino (b : Board) : Prop :=
  ∀ d : Domino, ∃ d' : Domino, d' ∈ b.dominoes

/-- Main theorem: It's possible to arrange all dominoes horizontally -/
theorem horizontal_arrangement_possible (b : Board) 
  (h_valid : is_valid_board b) (h_move : can_move_domino b) : 
  ∃ b' : Board, is_valid_board b' ∧ all_horizontal b' :=
sorry

end horizontal_arrangement_possible_l2912_291297


namespace milk_price_calculation_l2912_291256

/-- Calculates the final price of milk given wholesale price, markup percentage, and discount percentage -/
theorem milk_price_calculation (wholesale_price markup_percent discount_percent : ℝ) :
  wholesale_price = 4 →
  markup_percent = 25 →
  discount_percent = 5 →
  let retail_price := wholesale_price * (1 + markup_percent / 100)
  let final_price := retail_price * (1 - discount_percent / 100)
  final_price = 4.75 := by
  sorry

end milk_price_calculation_l2912_291256


namespace negation_of_existential_l2912_291280

theorem negation_of_existential (p : Prop) :
  (¬∃ (x : ℝ), x = Real.sin x) ↔ (∀ (x : ℝ), x ≠ Real.sin x) := by
  sorry

end negation_of_existential_l2912_291280


namespace parallel_vectors_imply_x_value_l2912_291206

/-- Given vectors a and b, and their linear combinations u and v, 
    prove that if u is parallel to v, then x = 1/2 -/
theorem parallel_vectors_imply_x_value 
  (a b u v : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (x, 1))
  (h3 : u = a + 2 • b)
  (h4 : v = 2 • a - b)
  (h5 : ∃ (k : ℝ), u = k • v)
  : x = 1/2 := by
  sorry

end parallel_vectors_imply_x_value_l2912_291206


namespace fraction_problem_l2912_291238

theorem fraction_problem (x y : ℚ) : 
  (x + 2) / (y + 1) = 1 → 
  (x + 4) / (y + 2) = 1/2 → 
  x / y = 5/4 := by
sorry

end fraction_problem_l2912_291238


namespace x_geq_y_l2912_291276

theorem x_geq_y (a : ℝ) : 2 * a * (a + 3) ≥ (a - 3) * (a + 3) := by
  sorry

end x_geq_y_l2912_291276


namespace wrapping_paper_area_l2912_291259

/-- The area of wrapping paper required for a rectangular box --/
theorem wrapping_paper_area
  (w v h : ℝ)
  (h_pos : 0 < h)
  (w_pos : 0 < w)
  (v_pos : 0 < v)
  (v_lt_w : v < w) :
  let paper_width := 3 * v
  let paper_length := w
  paper_width * paper_length = 3 * w * v :=
by sorry

end wrapping_paper_area_l2912_291259


namespace f_difference_nonnegative_l2912_291200

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem f_difference_nonnegative (x y : ℝ) :
  f x - f y ≥ 0 ↔ (x ≥ y ∧ x + y ≥ 6) ∨ (x ≤ y ∧ x + y ≤ 6) :=
by sorry

end f_difference_nonnegative_l2912_291200


namespace additional_calories_burnt_l2912_291217

def calories_per_hour : ℕ := 30

def calories_burnt (hours : ℕ) : ℕ := calories_per_hour * hours

theorem additional_calories_burnt : 
  calories_burnt 5 - calories_burnt 2 = 90 := by
  sorry

end additional_calories_burnt_l2912_291217


namespace bill_calculation_l2912_291223

theorem bill_calculation (a b c : ℝ) 
  (h1 : a - (b - c) = 11) 
  (h2 : a - b - c = 3) : 
  a - b = 7 := by
sorry

end bill_calculation_l2912_291223


namespace parallelogram_division_slope_l2912_291202

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ := (10, 30)
  v2 : ℝ × ℝ := (10, 80)
  v3 : ℝ × ℝ := (25, 125)
  v4 : ℝ × ℝ := (25, 75)

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- Predicate to check if a line divides a parallelogram into two congruent polygons -/
def divides_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- Theorem stating the slope of the line that divides the parallelogram -/
theorem parallelogram_division_slope (p : Parallelogram) (l : Line) :
  divides_into_congruent_polygons p l → l.slope = 24 / 7 :=
by sorry

end parallelogram_division_slope_l2912_291202


namespace vector_perpendicular_l2912_291295

/-- Given vectors m and n in ℝ², prove that if m + n is perpendicular to m - n, then t = -3 -/
theorem vector_perpendicular (t : ℝ) : 
  let m : Fin 2 → ℝ := ![t + 1, 1]
  let n : Fin 2 → ℝ := ![t + 2, 2]
  (m + n) • (m - n) = 0 → t = -3 := by
  sorry

end vector_perpendicular_l2912_291295


namespace total_books_l2912_291291

theorem total_books (sam_books joan_books : ℕ) 
  (h1 : sam_books = 110) 
  (h2 : joan_books = 102) : 
  sam_books + joan_books = 212 := by
  sorry

end total_books_l2912_291291


namespace paul_chickens_sold_to_neighbor_l2912_291227

/-- The number of chickens Paul sold to his neighbor -/
def chickens_sold_to_neighbor (initial_chickens : ℕ) (sold_to_customer : ℕ) (left_for_market : ℕ) : ℕ :=
  initial_chickens - sold_to_customer - left_for_market

theorem paul_chickens_sold_to_neighbor :
  chickens_sold_to_neighbor 80 25 43 = 12 := by
  sorry

end paul_chickens_sold_to_neighbor_l2912_291227


namespace problem_solution_l2912_291213

theorem problem_solution (a b : ℝ) (h1 : a - 2*b = 0) (h2 : b ≠ 0) :
  (b / (a - b) + 1) * (a^2 - b^2) / a^2 = 3/2 := by
  sorry

end problem_solution_l2912_291213


namespace digitSum5_125th_l2912_291279

/-- The sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence of natural numbers with digit sum 5 in ascending order --/
def digitSum5Seq : ℕ → ℕ := sorry

/-- The 125th number in the sequence of natural numbers with digit sum 5 --/
theorem digitSum5_125th :
  digitSum5Seq 125 = 41000 ∧ sumOfDigits (digitSum5Seq 125) = 5 := by sorry

end digitSum5_125th_l2912_291279


namespace tan_plus_cot_l2912_291289

theorem tan_plus_cot (α : Real) : 
  sinα - cosα = -Real.sqrt 5 / 2 → tanα + 1 / tanα = -8 := by
  sorry

end tan_plus_cot_l2912_291289


namespace danny_found_caps_l2912_291222

/-- Represents the number of bottle caps Danny had initially -/
def initial_caps : ℕ := 6

/-- Represents the total number of bottle caps Danny has now -/
def total_caps : ℕ := 28

/-- Represents the number of bottle caps Danny found at the park -/
def caps_found : ℕ := total_caps - initial_caps

theorem danny_found_caps : caps_found = 22 := by
  sorry

end danny_found_caps_l2912_291222


namespace even_function_order_l2912_291243

def f (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem even_function_order (b c : ℝ) 
  (h : ∀ x, f x b c = f (-x) b c) : 
  f 1 b c < f (-2) b c ∧ f (-2) b c < f 3 b c := by
  sorry

end even_function_order_l2912_291243


namespace cos_2x_derivative_l2912_291209

theorem cos_2x_derivative (x : ℝ) : 
  deriv (λ x => Real.cos (2 * x)) x = -2 * Real.sin (2 * x) := by
  sorry

end cos_2x_derivative_l2912_291209


namespace counterexample_exists_l2912_291242

theorem counterexample_exists (h : ∀ a b : ℝ, a > -b) : 
  ∃ a b : ℝ, a > -b ∧ (1/a) + (1/b) ≤ 0 := by sorry

end counterexample_exists_l2912_291242


namespace family_savings_by_end_of_2019_l2912_291216

/-- Proves that the family's savings by 31.12.2019 will be 1340840 rubles given their income, expenses, and initial savings. -/
theorem family_savings_by_end_of_2019 
  (income : ℕ) 
  (expenses : ℕ) 
  (initial_savings : ℕ) 
  (h1 : income = (55000 + 45000 + 10000 + 17400) * 4)
  (h2 : expenses = (40000 + 20000 + 5000 + 2000 + 2000) * 4)
  (h3 : initial_savings = 1147240) : 
  initial_savings + income - expenses = 1340840 :=
by sorry

end family_savings_by_end_of_2019_l2912_291216


namespace practice_time_for_second_recital_l2912_291253

/-- Represents the relationship between practice time and mistakes for a recital -/
structure Recital where
  practice_time : ℝ
  mistakes : ℝ

/-- The constant product of practice time and mistakes -/
def inverse_relation_constant (r : Recital) : ℝ :=
  r.practice_time * r.mistakes

theorem practice_time_for_second_recital
  (first_recital : Recital)
  (h1 : first_recital.practice_time = 5)
  (h2 : first_recital.mistakes = 12)
  (h3 : ∀ r : Recital, inverse_relation_constant r = inverse_relation_constant first_recital)
  (h4 : ∃ second_recital : Recital,
    (first_recital.mistakes + second_recital.mistakes) / 2 = 8) :
  ∃ second_recital : Recital, second_recital.practice_time = 15 := by
sorry

end practice_time_for_second_recital_l2912_291253


namespace geometric_series_first_term_l2912_291218

theorem geometric_series_first_term 
  (a r : ℝ) 
  (sum_condition : a / (1 - r) = 20) 
  (sum_squares_condition : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 := by
sorry

end geometric_series_first_term_l2912_291218


namespace distinct_roots_condition_root_condition_l2912_291262

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + k

-- Theorem for part 1
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic x k = 0 ∧ quadratic y k = 0) ↔ k < 1 :=
sorry

-- Theorem for part 2
theorem root_condition (m k : ℝ) :
  quadratic m k = 0 ∧ m^2 + 2*m = 2 → k = -2 :=
sorry

end distinct_roots_condition_root_condition_l2912_291262


namespace smallest_cookie_boxes_l2912_291212

theorem smallest_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(∃ (k : ℕ), 15 * m - 2 = 11 * k)) ∧ 
  (∃ (k : ℕ), 15 * n - 2 = 11 * k) := by
  sorry

end smallest_cookie_boxes_l2912_291212


namespace probability_between_R_and_S_l2912_291234

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PR and PQ = 8QR,
    the probability of a randomly selected point on PQ being between R and S is 5/8. -/
theorem probability_between_R_and_S (P Q R S : ℝ) : 
  P < R ∧ R < S ∧ S < Q ∧ Q - P = 4 * (R - P) ∧ Q - P = 8 * (Q - R) →
  (S - R) / (Q - P) = 5 / 8 := by
sorry

end probability_between_R_and_S_l2912_291234


namespace total_stamps_sold_l2912_291208

theorem total_stamps_sold (color_stamps : ℕ) (bw_stamps : ℕ) 
  (h1 : color_stamps = 578833) 
  (h2 : bw_stamps = 523776) : 
  color_stamps + bw_stamps = 1102609 := by
  sorry

end total_stamps_sold_l2912_291208


namespace equation_proof_l2912_291293

theorem equation_proof : 42 / (7 - 4/3) = 126/17 := by
  sorry

end equation_proof_l2912_291293


namespace larger_number_proof_l2912_291225

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 23) (h2 : Nat.lcm a b = 23 * 15 * 16) :
  max a b = 368 := by
sorry

end larger_number_proof_l2912_291225


namespace polygon_contains_circle_l2912_291282

/-- A convex polygon with width 1 -/
structure ConvexPolygon where
  width : ℝ
  width_eq_one : width = 1
  is_convex : Bool  -- This is a simplification, as convexity is more complex to define

/-- A circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Predicate to check if a circle is contained within a polygon -/
def containsCircle (p : ConvexPolygon) (c : Circle) : Prop :=
  sorry  -- The actual implementation would depend on how we represent polygons and circles

theorem polygon_contains_circle (M : ConvexPolygon) : 
  ∃ (c : Circle), c.radius ≥ 1/3 ∧ containsCircle M c := by
  sorry

#check polygon_contains_circle

end polygon_contains_circle_l2912_291282


namespace triangle_perimeter_l2912_291229

/-- Given a triangle PQR and parallel lines m_P, m_Q, m_R, find the perimeter of the triangle formed by these lines -/
theorem triangle_perimeter (PQ QR PR : ℝ) (m_P m_Q m_R : ℝ) : 
  PQ = 150 → QR = 270 → PR = 210 →
  m_P = 75 → m_Q = 60 → m_R = 30 →
  ∃ (perimeter : ℝ), abs (perimeter - 239.314) < 0.001 :=
by sorry


end triangle_perimeter_l2912_291229


namespace consecutive_sum_39_l2912_291244

theorem consecutive_sum_39 (n : ℕ) : 
  n + (n + 1) = 39 → n = 19 := by
sorry

end consecutive_sum_39_l2912_291244


namespace percent_relation_l2912_291241

theorem percent_relation (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : c = 0.1 * b) :
  b = 2 * a := by
  sorry

end percent_relation_l2912_291241


namespace quadratic_function_ordering_l2912_291272

/-- A quadratic function with the given properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  symmetry : ∀ x : ℝ, a * (2 + x)^2 + b * (2 + x) + c = a * (2 - x)^2 + b * (2 - x) + c

/-- The theorem stating the ordering of function values -/
theorem quadratic_function_ordering (f : QuadraticFunction) :
  f.a * 2^2 + f.b * 2 + f.c < f.a * 1^2 + f.b * 1 + f.c ∧
  f.a * 1^2 + f.b * 1 + f.c < f.a * 4^2 + f.b * 4 + f.c :=
sorry

end quadratic_function_ordering_l2912_291272


namespace march_greatest_drop_l2912_291284

-- Define the months
inductive Month
| January
| February
| March
| April
| May
| June

-- Define the price change for each month
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => -0.50
  | Month.February => 2.00
  | Month.March => -2.50
  | Month.April => 3.00
  | Month.May => -0.50
  | Month.June => -2.00

-- Define a function to check if a month has a price drop
def has_price_drop (m : Month) : Prop :=
  price_change m < 0

-- Define a function to compare price drops between two months
def greater_price_drop (m1 m2 : Month) : Prop :=
  has_price_drop m1 ∧ has_price_drop m2 ∧ price_change m1 < price_change m2

-- Theorem statement
theorem march_greatest_drop :
  ∀ m : Month, m ≠ Month.March → ¬(greater_price_drop m Month.March) :=
by sorry

end march_greatest_drop_l2912_291284


namespace tom_folder_purchase_l2912_291235

def remaining_money (initial_amount : ℕ) (folder_cost : ℕ) : ℕ :=
  initial_amount - (initial_amount / folder_cost) * folder_cost

theorem tom_folder_purchase : remaining_money 19 2 = 1 := by
  sorry

end tom_folder_purchase_l2912_291235


namespace square_root_fraction_sum_l2912_291246

theorem square_root_fraction_sum : 
  Real.sqrt (2/25 + 1/49 - 1/100) = 3/10 := by sorry

end square_root_fraction_sum_l2912_291246


namespace rainwater_solution_l2912_291292

/-- A tank collecting rainwater over three days -/
structure RainwaterTank where
  capacity : ℝ
  initialFill : ℝ
  day1Collection : ℝ
  day2Collection : ℝ
  day3Excess : ℝ

/-- The conditions of the rainwater collection problem -/
def rainProblem (tank : RainwaterTank) : Prop :=
  tank.capacity = 100 ∧
  tank.initialFill = 2/5 * tank.capacity ∧
  tank.day2Collection = tank.day1Collection + 5 ∧
  tank.initialFill + tank.day1Collection + tank.day2Collection = tank.capacity ∧
  tank.day3Excess = 25

/-- The theorem stating the solution to the rainwater problem -/
theorem rainwater_solution (tank : RainwaterTank) 
  (h : rainProblem tank) : tank.day1Collection = 27.5 := by
  sorry


end rainwater_solution_l2912_291292


namespace time_to_finish_book_l2912_291226

/-- Calculates the time needed to finish reading a book given the specified conditions -/
theorem time_to_finish_book 
  (total_chapters : ℕ) 
  (chapters_read : ℕ) 
  (time_for_read_chapters : ℝ) 
  (break_time : ℝ) 
  (h1 : total_chapters = 14) 
  (h2 : chapters_read = 4) 
  (h3 : time_for_read_chapters = 6) 
  (h4 : break_time = 1/6) : 
  let remaining_chapters := total_chapters - chapters_read
  let time_per_chapter := time_for_read_chapters / chapters_read
  let reading_time := time_per_chapter * remaining_chapters
  let total_breaks := remaining_chapters - 1
  let total_break_time := total_breaks * break_time
  reading_time + total_break_time = 33/2 := by
sorry

end time_to_finish_book_l2912_291226


namespace tv_show_episodes_l2912_291263

/-- Proves that a TV show with given conditions has 20 episodes per season in its first half -/
theorem tv_show_episodes (total_seasons : ℕ) (second_half_episodes : ℕ) (total_episodes : ℕ) :
  total_seasons = 10 →
  second_half_episodes = 25 →
  total_episodes = 225 →
  (total_seasons / 2 : ℕ) * second_half_episodes + (total_seasons / 2 : ℕ) * (total_episodes - (total_seasons / 2 : ℕ) * second_half_episodes) / (total_seasons / 2 : ℕ) = total_episodes →
  (total_episodes - (total_seasons / 2 : ℕ) * second_half_episodes) / (total_seasons / 2 : ℕ) = 20 :=
by sorry

end tv_show_episodes_l2912_291263


namespace no_integer_solution_l2912_291220

theorem no_integer_solution : ¬ ∃ (m n : ℤ), 5 * m^2 - 6 * m * n + 7 * n^2 = 2005 := by
  sorry

end no_integer_solution_l2912_291220


namespace two_digit_integer_property_l2912_291260

theorem two_digit_integer_property (a b k : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0) : 
  let n := 10 * a + b
  let m := 10 * b + a
  n = k * (a - b) → m = (k - 9) * (a - b) := by
sorry

end two_digit_integer_property_l2912_291260


namespace sequence_150th_term_l2912_291231

def sequence_term (n : ℕ) : ℕ := sorry

theorem sequence_150th_term : sequence_term 150 = 2280 := by sorry

end sequence_150th_term_l2912_291231


namespace line_plane_perpendicularity_l2912_291286

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel m n → 
  contains β n → 
  planePerp α β :=
sorry

end line_plane_perpendicularity_l2912_291286


namespace abs_sum_inequality_range_l2912_291221

theorem abs_sum_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) → a ≤ 4 := by
sorry

end abs_sum_inequality_range_l2912_291221


namespace gmat_test_percentage_l2912_291274

theorem gmat_test_percentage (S B N : ℝ) : 
  S = 70 → B = 60 → N = 5 → 100 - S + B - N = 85 :=
sorry

end gmat_test_percentage_l2912_291274


namespace apple_selling_price_l2912_291249

-- Define the cost price
def cost_price : ℚ := 17

-- Define the selling price as a function of the cost price
def selling_price (cp : ℚ) : ℚ := (5 / 6) * cp

-- Theorem stating that the selling price is 5/6 of the cost price
theorem apple_selling_price :
  selling_price cost_price = (5 / 6) * cost_price :=
by sorry

end apple_selling_price_l2912_291249


namespace ellipse_hyperbola_semi_axes_product_l2912_291204

theorem ellipse_hyperbola_semi_axes_product (a b : ℝ) : 
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a * b| = 2 * Real.sqrt 111 :=
by sorry

end ellipse_hyperbola_semi_axes_product_l2912_291204


namespace equivalence_condition_l2912_291236

theorem equivalence_condition (x y : ℕ) :
  (5 * x ≥ 7 * y) ↔
  (∃ a b c d : ℕ, x = a + 2*b + 3*c + 7*d ∧ y = b + 2*c + 5*d) :=
by sorry

end equivalence_condition_l2912_291236


namespace one_common_root_l2912_291288

def quadratic1 (x : ℝ) := x^2 + x - 6
def quadratic2 (x : ℝ) := x^2 - 7*x + 10

theorem one_common_root :
  ∃! r : ℝ, quadratic1 r = 0 ∧ quadratic2 r = 0 :=
sorry

end one_common_root_l2912_291288


namespace girls_total_distance_l2912_291277

/-- The number of laps run by the boys -/
def boys_laps : ℕ := 27

/-- The number of additional laps run by the girls compared to the boys -/
def extra_girls_laps : ℕ := 9

/-- The length of each boy's lap in miles -/
def boys_lap_length : ℚ := 3/4

/-- The length of the first type of girl's lap in miles -/
def girls_lap_length1 : ℚ := 3/4

/-- The length of the second type of girl's lap in miles -/
def girls_lap_length2 : ℚ := 7/8

/-- The total number of laps run by the girls -/
def girls_laps : ℕ := boys_laps + extra_girls_laps

/-- The number of laps of each type run by the girls -/
def girls_laps_each_type : ℕ := girls_laps / 2

theorem girls_total_distance :
  girls_laps_each_type * girls_lap_length1 + girls_laps_each_type * girls_lap_length2 = 29.25 := by
  sorry

end girls_total_distance_l2912_291277


namespace profit_achieved_l2912_291224

/-- The number of disks in a buy package -/
def buy_package : ℕ := 3

/-- The cost of a buy package in cents -/
def buy_cost : ℕ := 400

/-- The number of disks in a sell package -/
def sell_package : ℕ := 4

/-- The price of a sell package in cents -/
def sell_price : ℕ := 600

/-- The target profit in cents -/
def target_profit : ℕ := 15000

/-- The minimum number of disks to be sold to achieve the target profit -/
def min_disks_to_sell : ℕ := 883

theorem profit_achieved : 
  ∃ (n : ℕ), n ≥ min_disks_to_sell ∧ 
  (n * sell_price / sell_package - n * buy_cost / buy_package) ≥ target_profit ∧
  ∀ (m : ℕ), m < min_disks_to_sell → 
  (m * sell_price / sell_package - m * buy_cost / buy_package) < target_profit :=
sorry

end profit_achieved_l2912_291224


namespace inscribed_sphere_surface_area_l2912_291233

theorem inscribed_sphere_surface_area (cube_edge : ℝ) (sphere_area : ℝ) :
  cube_edge = 2 →
  sphere_area = 4 * Real.pi →
  sphere_area = (4 : ℝ) * Real.pi * (cube_edge / 2) ^ 2 :=
by sorry

end inscribed_sphere_surface_area_l2912_291233


namespace triangle_interior_angle_l2912_291250

theorem triangle_interior_angle (a b : ℝ) (ha : a = 110) (hb : b = 120) : 
  ∃ x : ℝ, x = 50 ∧ x + (360 - (a + b)) = 180 := by
  sorry

end triangle_interior_angle_l2912_291250


namespace harmonic_interval_k_range_l2912_291232

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

def is_harmonic_interval (k a b : ℝ) : Prop :=
  a ≤ b ∧ a ≥ 1 ∧ b ≥ 1 ∧
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∧
  f a = k * a ∧ f b = k * b

theorem harmonic_interval_k_range :
  {k : ℝ | ∃ a b, is_harmonic_interval k a b} = Set.Ioo 2 3 := by sorry

end harmonic_interval_k_range_l2912_291232


namespace quadratic_inequality_range_l2912_291230

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
sorry

end quadratic_inequality_range_l2912_291230


namespace a_6_equals_one_half_l2912_291298

def a (n : ℕ+) : ℚ := (3 * n.val - 2) / (2 ^ (n.val - 1))

theorem a_6_equals_one_half : a 6 = 1 / 2 := by sorry

end a_6_equals_one_half_l2912_291298


namespace train_speed_calculation_train_speed_result_l2912_291210

/-- Calculates the speed of a train given its length, the time it takes to pass a walking man, and the man's speed. -/
theorem train_speed_calculation (train_length : ℝ) (passing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_ms := relative_speed + man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  train_speed_kmh

/-- The speed of the train is approximately 63.0036 km/hr given the specified conditions. -/
theorem train_speed_result :
  ∃ ε > 0, |train_speed_calculation 900 53.99568034557235 3 - 63.0036| < ε :=
sorry

end train_speed_calculation_train_speed_result_l2912_291210


namespace consecutive_odd_power_sum_divisible_l2912_291265

-- Define consecutive odd numbers
def ConsecutiveOddNumbers (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = 2*k + 1 ∧ b = 2*k + 3

-- Define divisibility
def Divides (d n : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Theorem statement
theorem consecutive_odd_power_sum_divisible (a b : ℕ) :
  ConsecutiveOddNumbers a b → Divides (a + b) (a^b + b^a) :=
by
  sorry

end consecutive_odd_power_sum_divisible_l2912_291265


namespace total_cars_is_43_l2912_291239

/-- The number of cars owned by each person -/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  carol : ℕ
  susan : ℕ
  erica : ℕ

/-- Conditions for car ownership -/
def validCarOwnership (c : CarOwnership) : Prop :=
  c.lindsey = c.cathy + 4 ∧
  c.susan = c.carol - 2 ∧
  c.carol = 2 * c.cathy ∧
  c.erica = c.lindsey + (c.lindsey / 4) ∧
  c.cathy = 5

/-- The total number of cars owned by all people -/
def totalCars (c : CarOwnership) : ℕ :=
  c.cathy + c.lindsey + c.carol + c.susan + c.erica

/-- Theorem stating that the total number of cars is 43 -/
theorem total_cars_is_43 (c : CarOwnership) (h : validCarOwnership c) : totalCars c = 43 := by
  sorry

end total_cars_is_43_l2912_291239


namespace exam_marks_category_c_l2912_291270

theorem exam_marks_category_c (total_candidates : ℕ) 
                               (category_a_count : ℕ) 
                               (category_b_count : ℕ) 
                               (category_c_count : ℕ) 
                               (category_a_avg : ℕ) 
                               (category_b_avg : ℕ) 
                               (category_c_avg : ℕ) : 
  total_candidates = 80 →
  category_a_count = 30 →
  category_b_count = 25 →
  category_c_count = 25 →
  category_a_avg = 35 →
  category_b_avg = 42 →
  category_c_avg = 46 →
  category_c_count * category_c_avg = 1150 :=
by sorry

end exam_marks_category_c_l2912_291270


namespace expression_value_at_negative_one_l2912_291271

theorem expression_value_at_negative_one :
  let x : ℤ := -1
  (x^2 + 5*x - 6) = -10 := by sorry

end expression_value_at_negative_one_l2912_291271


namespace total_shells_l2912_291258

/-- The number of shells each person has -/
structure ShellCounts where
  david : ℕ
  mia : ℕ
  ava : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def shell_problem (counts : ShellCounts) : Prop :=
  counts.david = 15 ∧
  counts.mia = 4 * counts.david ∧
  counts.ava = counts.mia + 20 ∧
  counts.alice = counts.ava / 2

/-- The theorem to prove -/
theorem total_shells (counts : ShellCounts) : 
  shell_problem counts → counts.david + counts.mia + counts.ava + counts.alice = 195 := by
  sorry

end total_shells_l2912_291258


namespace venus_meal_cost_is_35_l2912_291281

/-- The cost per meal at Venus Hall -/
def venus_meal_cost : ℚ := 35

/-- The room rental cost at Caesar's -/
def caesars_rental : ℚ := 800

/-- The cost per meal at Caesar's -/
def caesars_meal_cost : ℚ := 30

/-- The room rental cost at Venus Hall -/
def venus_rental : ℚ := 500

/-- The number of guests at which the total costs are equal -/
def num_guests : ℚ := 60

theorem venus_meal_cost_is_35 :
  caesars_rental + caesars_meal_cost * num_guests =
  venus_rental + venus_meal_cost * num_guests := by
  sorry

end venus_meal_cost_is_35_l2912_291281


namespace graph_shift_l2912_291248

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the transformation
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x => g x - 3

-- Theorem statement
theorem graph_shift (x y : ℝ) : 
  y = transform g x ↔ y + 3 = g x := by sorry

end graph_shift_l2912_291248


namespace axes_of_symmetry_coincide_l2912_291219

/-- Two quadratic functions with their coefficients -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  r : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  f₁ : QuadraticFunction
  f₂ : QuadraticFunction
  A : Point
  B : Point
  p₁_positive : f₁.p > 0
  p₂_negative : f₂.p < 0
  distinct_intersections : A ≠ B
  intersections_on_curves : 
    A.y = f₁.p * A.x^2 + f₁.q * A.x + f₁.r ∧
    A.y = f₂.p * A.x^2 + f₂.q * A.x + f₂.r ∧
    B.y = f₁.p * B.x^2 + f₁.q * B.x + f₁.r ∧
    B.y = f₂.p * B.x^2 + f₂.q * B.x + f₂.r
  tangents_form_cyclic_quad : True  -- This is a placeholder for the cyclic quadrilateral condition

/-- The main theorem stating that the axes of symmetry coincide -/
theorem axes_of_symmetry_coincide (setup : ProblemSetup) : 
  setup.f₁.q / setup.f₁.p = setup.f₂.q / setup.f₂.p := by
  sorry

end axes_of_symmetry_coincide_l2912_291219


namespace popsicle_sticks_problem_l2912_291296

theorem popsicle_sticks_problem (steve sid sam : ℕ) : 
  sid = 2 * steve →
  sam = 3 * sid →
  steve + sid + sam = 108 →
  steve = 12 := by
sorry

end popsicle_sticks_problem_l2912_291296


namespace batsman_average_l2912_291283

/-- Given a batsman whose average increases by 3 after scoring 66 runs in the 17th inning,
    his new average after the 17th inning is 18. -/
theorem batsman_average (prev_average : ℝ) : 
  (16 * prev_average + 66) / 17 = prev_average + 3 → prev_average + 3 = 18 := by
sorry

end batsman_average_l2912_291283


namespace same_solution_value_l2912_291211

theorem same_solution_value (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 5 = 1 ∧ c * x + 15 = -5) ↔ c = 15 := by
sorry

end same_solution_value_l2912_291211


namespace sin_cos_identity_l2912_291255

theorem sin_cos_identity : Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
                           Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end sin_cos_identity_l2912_291255


namespace percentage_fraction_difference_l2912_291264

theorem percentage_fraction_difference : 
  (85 / 100 * 40) - (4 / 5 * 25) = 14 := by sorry

end percentage_fraction_difference_l2912_291264


namespace average_tv_watching_three_weeks_l2912_291215

def tv_watching (week1 week2 week3 : ℕ) : ℕ := week1 + week2 + week3

def average_tv_watching (total_hours num_weeks : ℕ) : ℚ :=
  (total_hours : ℚ) / (num_weeks : ℚ)

theorem average_tv_watching_three_weeks :
  let week1 : ℕ := 10
  let week2 : ℕ := 8
  let week3 : ℕ := 12
  let total_hours := tv_watching week1 week2 week3
  let num_weeks : ℕ := 3
  average_tv_watching total_hours num_weeks = 10 := by
  sorry

end average_tv_watching_three_weeks_l2912_291215


namespace walnut_trees_after_planting_l2912_291207

theorem walnut_trees_after_planting 
  (initial_trees : ℕ) 
  (new_trees : ℕ) 
  (h1 : initial_trees = 107) 
  (h2 : new_trees = 104) : 
  initial_trees + new_trees = 211 := by
  sorry

end walnut_trees_after_planting_l2912_291207


namespace nail_decoration_time_l2912_291268

def base_coat_time : ℕ := 20
def paint_coat_time : ℕ := 20
def glitter_coat_time : ℕ := 20
def drying_time : ℕ := 20
def pattern_time : ℕ := 40

def total_decoration_time : ℕ :=
  base_coat_time + drying_time +
  paint_coat_time + drying_time +
  glitter_coat_time + drying_time +
  pattern_time

theorem nail_decoration_time :
  total_decoration_time = 160 :=
by sorry

end nail_decoration_time_l2912_291268


namespace snail_count_l2912_291247

/-- The number of snails gotten rid of in Centerville -/
def snails_removed : ℕ := 3482

/-- The number of snails remaining in Centerville -/
def snails_remaining : ℕ := 8278

/-- The original number of snails in Centerville -/
def original_snails : ℕ := snails_removed + snails_remaining

theorem snail_count : original_snails = 11760 := by
  sorry

end snail_count_l2912_291247


namespace sum_18_29_base4_l2912_291278

/-- Converts a number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_18_29_base4 :
  toBase4 (18 + 29) = [2, 3, 3] :=
sorry

end sum_18_29_base4_l2912_291278


namespace inequality_proof_l2912_291214

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c := by
  sorry

end inequality_proof_l2912_291214


namespace candy_sharing_theorem_l2912_291273

/-- Represents the amount of candy each person has initially -/
structure CandyDistribution where
  hugh : ℕ
  tommy : ℕ
  melany : ℕ

/-- Calculates the amount of candy each person gets when shared equally -/
def equalShare (dist : CandyDistribution) : ℕ :=
  (dist.hugh + dist.tommy + dist.melany) / 3

/-- Theorem: When Hugh has 8 pounds, Tommy has 6 pounds, and Melany has 7 pounds of candy,
    sharing equally results in each person having 7 pounds of candy -/
theorem candy_sharing_theorem (dist : CandyDistribution) 
  (h1 : dist.hugh = 8) 
  (h2 : dist.tommy = 6) 
  (h3 : dist.melany = 7) : 
  equalShare dist = 7 := by
  sorry

end candy_sharing_theorem_l2912_291273


namespace wrench_hammer_weight_ratio_l2912_291287

/-- Given that hammers and wrenches have uniform weights, prove that if the total weight of 2 hammers
    and 2 wrenches is 1/3 of the weight of 8 hammers and 5 wrenches, then the weight of one wrench
    is 2 times the weight of one hammer. -/
theorem wrench_hammer_weight_ratio 
  (h : ℝ) -- weight of one hammer
  (w : ℝ) -- weight of one wrench
  (h_pos : h > 0) -- hammer weight is positive
  (w_pos : w > 0) -- wrench weight is positive
  (weight_ratio : 2 * h + 2 * w = (1 / 3) * (8 * h + 5 * w)) -- given condition
  : w = 2 * h := by
  sorry

end wrench_hammer_weight_ratio_l2912_291287


namespace integer_roots_quadratic_l2912_291299

theorem integer_roots_quadratic (n : ℤ) : 
  (∃ x y : ℤ, x^2 + (n+1)*x + 2*n - 1 = 0 ∧ y^2 + (n+1)*y + 2*n - 1 = 0) → 
  (n = 1 ∨ n = 5) := by
sorry

end integer_roots_quadratic_l2912_291299


namespace surface_area_comparison_l2912_291254

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  
/-- Represents a point on the parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a chord of the parabola -/
structure Chord where
  p1 : ParabolaPoint
  p2 : ParabolaPoint

/-- Represents the projection of a chord onto the directrix -/
def projection (c : Chord) : ℝ := sorry

/-- Surface area formed by rotating a chord around the directrix -/
def surfaceAreaRotation (c : Chord) : ℝ := sorry

/-- Surface area of a sphere with given diameter -/
def surfaceAreaSphere (diameter : ℝ) : ℝ := sorry

/-- Theorem stating that the surface area of rotation is greater than or equal to
    the surface area of the sphere formed by the projection -/
theorem surface_area_comparison
  (para : Parabola) (c : Chord) 
  (h1 : c.p1.y^2 = 2 * para.p * c.p1.x)
  (h2 : c.p2.y^2 = 2 * para.p * c.p2.x)
  (h3 : c.p1.x + c.p2.x = 2 * para.p) -- chord passes through focus
  : surfaceAreaRotation c ≥ surfaceAreaSphere (projection c) := by
  sorry

end surface_area_comparison_l2912_291254


namespace urn_has_eleven_marbles_l2912_291275

/-- Represents the number of marbles in an urn -/
structure Urn where
  green : ℕ
  yellow : ℕ

/-- The conditions of the marble problem -/
def satisfies_conditions (u : Urn) : Prop :=
  (4 * (u.green - 3) = u.green + u.yellow - 3) ∧
  (3 * u.green = u.green + u.yellow - 4)

/-- The theorem stating that an urn satisfying the conditions has 11 marbles -/
theorem urn_has_eleven_marbles (u : Urn) 
  (h : satisfies_conditions u) : u.green + u.yellow = 11 := by
  sorry

#check urn_has_eleven_marbles

end urn_has_eleven_marbles_l2912_291275


namespace increments_theorem_l2912_291267

/-- The function z(x, y) = xy -/
def z (x y : ℝ) : ℝ := x * y

/-- The initial point M₀ -/
def M₀ : ℝ × ℝ := (1, 2)

/-- The point M₁ -/
def M₁ : ℝ × ℝ := (1.1, 2)

/-- The point M₂ -/
def M₂ : ℝ × ℝ := (1, 1.9)

/-- The point M₃ -/
def M₃ : ℝ × ℝ := (1.1, 2.2)

/-- The increment of z from M₀ to another point -/
def increment (M : ℝ × ℝ) : ℝ := z M.1 M.2 - z M₀.1 M₀.2

theorem increments_theorem :
  increment M₁ = 0.2 ∧ increment M₂ = -0.1 ∧ increment M₃ = 0.42 := by
  sorry

end increments_theorem_l2912_291267


namespace bus_driver_worked_69_hours_l2912_291290

/-- Represents the payment structure and total compensation for a bus driver --/
structure BusDriverPayment where
  regular_rate : ℝ
  overtime_rate : ℝ
  double_overtime_rate : ℝ
  total_compensation : ℝ

/-- Calculates the total hours worked by a bus driver given their payment structure and total compensation --/
def calculate_total_hours (payment : BusDriverPayment) : ℕ :=
  sorry

/-- Theorem stating that given the specific payment structure and total compensation, the bus driver worked 69 hours --/
theorem bus_driver_worked_69_hours : 
  let payment := BusDriverPayment.mk 14 18.90 24.50 1230
  calculate_total_hours payment = 69 := by
  sorry

end bus_driver_worked_69_hours_l2912_291290


namespace sum_of_fractions_in_base_10_l2912_291269

/-- Convert a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Express a fraction in base 10 given numerator and denominator in different bases -/
def fractionToBase10 (num : ℕ) (num_base : ℕ) (den : ℕ) (den_base : ℕ) : ℚ := sorry

/-- Main theorem: The integer part of the sum of the given fractions in base 10 is 29 -/
theorem sum_of_fractions_in_base_10 : 
  ⌊(fractionToBase10 254 8 13 4 + fractionToBase10 132 5 22 3)⌋ = 29 := by sorry

end sum_of_fractions_in_base_10_l2912_291269


namespace fish_to_rice_value_l2912_291237

-- Define the trade rates
def fish_to_bread_rate : ℚ := 2 / 3
def bread_to_rice_rate : ℚ := 4

-- Theorem statement
theorem fish_to_rice_value :
  fish_to_bread_rate * bread_to_rice_rate = 8 / 3 :=
by sorry

end fish_to_rice_value_l2912_291237


namespace bank_account_final_balance_l2912_291228

-- Define the initial balance and transactions
def initial_balance : ℚ := 500
def first_withdrawal : ℚ := 200
def second_withdrawal_ratio : ℚ := 1/3
def first_deposit_ratio : ℚ := 1/5
def second_deposit_ratio : ℚ := 3/7

-- Define the theorem
theorem bank_account_final_balance :
  let balance_after_first_withdrawal := initial_balance - first_withdrawal
  let balance_after_second_withdrawal := balance_after_first_withdrawal * (1 - second_withdrawal_ratio)
  let balance_after_first_deposit := balance_after_second_withdrawal * (1 + first_deposit_ratio)
  let final_balance := balance_after_first_deposit / (1 - second_deposit_ratio)
  final_balance = 420 := by
  sorry

end bank_account_final_balance_l2912_291228


namespace grid_coloring_4x2011_l2912_291266

/-- Represents the number of ways to color a 4 × n grid with the given constraints -/
def coloringWays (n : ℕ) : ℕ :=
  64 * 3^(2*n)

/-- The problem statement -/
theorem grid_coloring_4x2011 :
  coloringWays 2011 = 64 * 3^4020 :=
by sorry

end grid_coloring_4x2011_l2912_291266


namespace joined_right_triangles_square_areas_l2912_291240

theorem joined_right_triangles_square_areas 
  (AB BC CD : ℝ) 
  (h_AB : AB^2 = 49) 
  (h_BC : BC^2 = 25) 
  (h_CD : CD^2 = 64) 
  (h_ABC_right : AB^2 + BC^2 = AC^2) 
  (h_ACD_right : CD^2 + AD^2 = AC^2) : 
  AD^2 = 10 := by
sorry

end joined_right_triangles_square_areas_l2912_291240


namespace consecutive_integers_sum_l2912_291201

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 380) : 
  x + (x + 1) = 39 := by
  sorry

end consecutive_integers_sum_l2912_291201


namespace quadratic_linear_relationship_l2912_291245

/-- Given a quadratic function y₁ and a linear function y₂, prove the relationship between b and c -/
theorem quadratic_linear_relationship (a b c : ℝ) : 
  let y₁ := fun x => (x + 2*a) * (x - 2*b)
  let y₂ := fun x => -x + 2*b
  let y := fun x => y₁ x + y₂ x
  a + 2 = b → 
  y c = 0 → 
  (c = 5 - 2*b ∨ c = 2*b) := by sorry

end quadratic_linear_relationship_l2912_291245


namespace blueberry_trade_l2912_291203

/-- The number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 7

/-- The number of containers of blueberries that can be exchanged for zucchinis -/
def containers_per_exchange : ℕ := 7

/-- The number of zucchinis received in one exchange -/
def zucchinis_per_exchange : ℕ := 3

/-- The total number of zucchinis Natalie wants to trade for -/
def target_zucchinis : ℕ := 63

/-- The number of bushes needed to trade for the target number of zucchinis -/
def bushes_needed : ℕ := 21

theorem blueberry_trade :
  bushes_needed * containers_per_bush * zucchinis_per_exchange =
  target_zucchinis * containers_per_exchange :=
by sorry

end blueberry_trade_l2912_291203


namespace product_of_solutions_abs_eq_l2912_291205

theorem product_of_solutions_abs_eq : ∃ (a b : ℝ), 
  (∀ x : ℝ, (|x| = 3 * (|x| - 4)) ↔ (x = a ∨ x = b)) ∧ (a * b = -36) := by
  sorry

end product_of_solutions_abs_eq_l2912_291205


namespace equation_transformation_l2912_291285

theorem equation_transformation (x : ℝ) : 3*(x+1) - 5*(1-x) = 3*x + 3 - 5 + 5*x := by
  sorry

end equation_transformation_l2912_291285


namespace smallest_multiple_1_to_10_l2912_291261

theorem smallest_multiple_1_to_10 : ∀ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) → n ≥ 2520 :=
by
  sorry

end smallest_multiple_1_to_10_l2912_291261


namespace not_integer_fraction_l2912_291294

theorem not_integer_fraction (a b : ℤ) : ¬ (∃ (k : ℤ), (a^2 + b^2) = k * (a^2 - b^2)) :=
by
  sorry

end not_integer_fraction_l2912_291294


namespace factor_expression_l2912_291257

theorem factor_expression (m n x y : ℝ) : m * (x - y) + n * (y - x) = (x - y) * (m - n) := by
  sorry

end factor_expression_l2912_291257


namespace simplify_fraction_1_simplify_fraction_2_simplify_fraction_3_l2912_291252

-- Part 1
theorem simplify_fraction_1 (x : ℝ) (h : x ≠ 1) :
  (3 * x + 2) / (x - 1) - 5 / (x - 1) = 3 :=
by sorry

-- Part 2
theorem simplify_fraction_2 (a : ℝ) (h : a ≠ 3) :
  (a^2) / (a^2 - 6*a + 9) / (a / (a - 3)) = a / (a - 3) :=
by sorry

-- Part 3
theorem simplify_fraction_3 (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ -4) :
  (x - 4) / (x + 3) / (x - 3 - 7 / (x + 3)) = 1 / (x + 4) :=
by sorry

end simplify_fraction_1_simplify_fraction_2_simplify_fraction_3_l2912_291252


namespace function_decreasing_implies_a_range_l2912_291251

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) :=
sorry

end function_decreasing_implies_a_range_l2912_291251
