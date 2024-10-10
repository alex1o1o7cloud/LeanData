import Mathlib

namespace pyramid_height_specific_l3937_393715

/-- Represents a pyramid with a square base and identical triangular faces. -/
structure Pyramid where
  base_area : ℝ
  face_area : ℝ

/-- The height of a pyramid given its base area and face area. -/
def pyramid_height (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating that a pyramid with base area 1440 and face area 840 has height 40. -/
theorem pyramid_height_specific : 
  ∀ (p : Pyramid), p.base_area = 1440 ∧ p.face_area = 840 → pyramid_height p = 40 := by
  sorry

end pyramid_height_specific_l3937_393715


namespace cost_of_field_trip_l3937_393793

def field_trip_cost (grandma_contribution : ℝ) (candy_bar_price : ℝ) (candy_bars_to_sell : ℕ) : ℝ :=
  grandma_contribution + candy_bar_price * (candy_bars_to_sell : ℝ)

theorem cost_of_field_trip :
  field_trip_cost 250 1.25 188 = 485 := by
  sorry

end cost_of_field_trip_l3937_393793


namespace zero_existence_l3937_393789

theorem zero_existence (f : ℝ → ℝ) (hf : Continuous f) 
  (h0 : f 0 = -3) (h1 : f 1 = 6) (h3 : f 3 = -5) :
  ∃ x₁ ∈ Set.Ioo 0 1, f x₁ = 0 ∧ ∃ x₂ ∈ Set.Ioo 1 3, f x₂ = 0 := by
  sorry

end zero_existence_l3937_393789


namespace arithmetic_sequence_common_difference_l3937_393705

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- arithmetic sequence
  (p q : ℕ)    -- indices
  (h1 : a p = 4)
  (h2 : a q = 2)
  (h3 : p = 4 + q)
  (h4 : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) :  -- arithmetic sequence property
  ∃ d : ℝ, d = 1/2 ∧ ∀ n : ℕ, a (n + 1) - a n = d := by
sorry

end arithmetic_sequence_common_difference_l3937_393705


namespace negation_of_existence_proposition_l3937_393798

theorem negation_of_existence_proposition :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end negation_of_existence_proposition_l3937_393798


namespace inequality_proof_l3937_393701

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_sum : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end inequality_proof_l3937_393701


namespace caitlin_age_is_24_l3937_393749

/-- The age of Aunt Anna in years -/
def aunt_anna_age : ℕ := 45

/-- The age of Brianna in years -/
def brianna_age : ℕ := (2 * aunt_anna_age) / 3

/-- The age difference between Brianna and Caitlin in years -/
def age_difference : ℕ := 6

/-- The age of Caitlin in years -/
def caitlin_age : ℕ := brianna_age - age_difference

/-- Theorem stating Caitlin's age -/
theorem caitlin_age_is_24 : caitlin_age = 24 := by sorry

end caitlin_age_is_24_l3937_393749


namespace expression_value_l3937_393725

theorem expression_value : 
  let a : ℚ := 1/2
  (2 * a⁻¹ + a⁻¹ / 2) / a = 10 := by sorry

end expression_value_l3937_393725


namespace bug_probability_7_l3937_393740

/-- Probability of a bug being at the starting vertex of a regular tetrahedron after n steps -/
def bug_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | m + 1 => (1 / 3) * (1 - bug_probability m)

/-- The probability of the bug being at the starting vertex after 7 steps is 182/729 -/
theorem bug_probability_7 : bug_probability 7 = 182 / 729 := by
  sorry

end bug_probability_7_l3937_393740


namespace real_roots_range_l3937_393728

theorem real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*m*x + 4*m^2 + 2*m + 3 = 0 ∨ x^2 + (2*m + 1)*x + m^2 = 0) ↔ 
  (m ≤ -3/2 ∨ m ≥ -1/4) :=
sorry

end real_roots_range_l3937_393728


namespace tangent_line_parabola_hyperbola_eccentricity_l3937_393782

/-- Given a line y = kx - 1 tangent to the parabola x² = 8y, 
    the eccentricity of the hyperbola x² - k²y² = 1 is equal to √3 -/
theorem tangent_line_parabola_hyperbola_eccentricity :
  ∀ k : ℝ,
  (∃ x y : ℝ, y = k * x - 1 ∧ x^2 = 8 * y ∧ 
   ∀ x' y' : ℝ, y' = k * x' - 1 → x'^2 ≠ 8 * y' ∨ (x' = x ∧ y' = y)) →
  Real.sqrt 3 = (Real.sqrt (1 + (1 / k^2))) / 1 :=
by sorry

end tangent_line_parabola_hyperbola_eccentricity_l3937_393782


namespace tom_candy_left_l3937_393750

def initial_candy : ℕ := 2
def friend_candy : ℕ := 7
def bought_candy : ℕ := 10

def total_candy : ℕ := initial_candy + friend_candy + bought_candy

def candy_left : ℕ := total_candy - (total_candy / 2)

theorem tom_candy_left : candy_left = 10 := by sorry

end tom_candy_left_l3937_393750


namespace poster_cost_l3937_393784

theorem poster_cost (initial_money : ℕ) (book1_cost : ℕ) (book2_cost : ℕ) (num_posters : ℕ) :
  initial_money = 20 →
  book1_cost = 8 →
  book2_cost = 4 →
  num_posters = 2 →
  initial_money - (book1_cost + book2_cost) = num_posters * (initial_money - (book1_cost + book2_cost)) / num_posters :=
by
  sorry

end poster_cost_l3937_393784


namespace total_wheat_mass_l3937_393772

def wheat_weights : List Float := [90, 91, 91.5, 89, 91.2, 91.3, 89.7, 88.8, 91.8, 91.1]

theorem total_wheat_mass :
  wheat_weights.sum = 905.4 := by
  sorry

end total_wheat_mass_l3937_393772


namespace max_area_of_remaining_rectangle_l3937_393719

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the configuration of rectangles in the square -/
structure SquareConfiguration where
  sideLength : ℝ
  rect1 : Rectangle
  rect2 : Rectangle
  rectR : Rectangle

/-- The theorem statement -/
theorem max_area_of_remaining_rectangle (config : SquareConfiguration) :
  config.sideLength ≥ 4 →
  config.rect1.width = 2 ∧ config.rect1.height = 4 →
  config.rect2.width = 2 ∧ config.rect2.height = 2 →
  config.rectR.area ≤ config.sideLength^2 - 12 ∧
  (config.sideLength = 4 → config.rectR.area = 4) :=
by sorry

end max_area_of_remaining_rectangle_l3937_393719


namespace monic_quartic_polynomial_value_l3937_393742

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) :
  MonicQuarticPolynomial p →
  p 2 = 7 →
  p 3 = 12 →
  p 4 = 19 →
  p 5 = 28 →
  p 6 = 63 := by
  sorry

end monic_quartic_polynomial_value_l3937_393742


namespace gymnasium_doubles_players_l3937_393743

theorem gymnasium_doubles_players (total_tables : ℕ) 
  (h1 : total_tables = 13) 
  (h2 : ∀ x y : ℕ, x + y = total_tables → 4 * x - 2 * y = 4 → 4 * x = 20) :
  ∃ x y : ℕ, x + y = total_tables ∧ 4 * x - 2 * y = 4 ∧ 4 * x = 20 :=
sorry

end gymnasium_doubles_players_l3937_393743


namespace game_prime_exists_l3937_393726

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem game_prime_exists : 
  ∃ p : ℕ, 
    is_prime p ∧ 
    ∃ (a b c d : ℕ), 
      p = a * 1000 + b * 100 + c * 10 + d ∧
      a ∈ ({4, 7, 8} : Set ℕ) ∧
      b ∈ ({4, 5, 9} : Set ℕ) ∧
      c ∈ ({1, 2, 3} : Set ℕ) ∧
      d < 10 ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      p = 8923 :=
by
  sorry

end game_prime_exists_l3937_393726


namespace function_is_constant_l3937_393763

/-- A function f: ℚ → ℝ satisfying |f(x) - f(y)| ≤ (x - y)² for all x, y ∈ ℚ is constant. -/
theorem function_is_constant (f : ℚ → ℝ) 
  (h : ∀ x y : ℚ, |f x - f y| ≤ (x - y)^2) : 
  ∃ c : ℝ, ∀ x : ℚ, f x = c :=
sorry

end function_is_constant_l3937_393763


namespace product_loop_result_l3937_393777

def product_loop (i : ℕ) : ℕ :=
  if i < 11 then 1 else i * product_loop (i - 1)

theorem product_loop_result :
  product_loop 12 = 132 :=
by sorry

end product_loop_result_l3937_393777


namespace b_most_suitable_l3937_393744

/-- Represents a candidate in the competition -/
structure Candidate where
  name : String
  average_score : ℝ
  variance : ℝ

/-- The set of all candidates -/
def candidates : Set Candidate :=
  { ⟨"A", 92.5, 3.4⟩, ⟨"B", 92.5, 2.1⟩, ⟨"C", 92.5, 2.5⟩, ⟨"D", 92.5, 2.7⟩ }

/-- Definition of the most suitable candidate -/
def most_suitable (c : Candidate) : Prop :=
  c ∈ candidates ∧
  ∀ d ∈ candidates, c.variance ≤ d.variance

/-- Theorem stating that B is the most suitable candidate -/
theorem b_most_suitable :
  ∃ c ∈ candidates, c.name = "B" ∧ most_suitable c := by
  sorry

end b_most_suitable_l3937_393744


namespace race_theorem_l3937_393765

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance_run : ℝ → ℝ

/-- The race setup -/
structure Race where
  petya : Runner
  kolya : Runner
  vasya : Runner
  race_distance : ℝ

def Race.valid (r : Race) : Prop :=
  r.race_distance = 100 ∧
  r.petya.speed > 0 ∧ r.kolya.speed > 0 ∧ r.vasya.speed > 0 ∧
  r.petya.distance_run 0 = 0 ∧ r.kolya.distance_run 0 = 0 ∧ r.vasya.distance_run 0 = 0 ∧
  ∀ t, r.petya.distance_run t = r.petya.speed * t ∧
       r.kolya.distance_run t = r.kolya.speed * t ∧
       r.vasya.distance_run t = r.vasya.speed * t

def Race.petya_finishes_first (r : Race) : Prop :=
  ∃ t, r.petya.distance_run t = r.race_distance ∧
       r.kolya.distance_run t < r.race_distance ∧
       r.vasya.distance_run t < r.race_distance

def Race.half_distance_condition (r : Race) : Prop :=
  ∃ t, r.petya.distance_run t = r.race_distance / 2 ∧
       r.kolya.distance_run t + r.vasya.distance_run t = 85

theorem race_theorem (r : Race) (h_valid : r.valid) (h_first : r.petya_finishes_first)
    (h_half : r.half_distance_condition) :
    ∃ t, r.petya.distance_run t = r.race_distance ∧
         2 * r.race_distance - (r.kolya.distance_run t + r.vasya.distance_run t) = 30 := by
  sorry

end race_theorem_l3937_393765


namespace prime_sum_special_equation_l3937_393751

theorem prime_sum_special_equation (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → q^5 - 2*p^2 = 1 → p + q = 14 := by
  sorry

end prime_sum_special_equation_l3937_393751


namespace last_year_production_l3937_393779

/-- The number of eggs produced by farms in Douglas County --/
structure EggProduction where
  lastYear : ℕ
  thisYear : ℕ
  increase : ℕ

/-- Theorem stating the relationship between this year's and last year's egg production --/
theorem last_year_production (e : EggProduction) 
  (h1 : e.thisYear = 4636)
  (h2 : e.increase = 3220)
  (h3 : e.thisYear = e.lastYear + e.increase) :
  e.lastYear = 1416 := by
  sorry

end last_year_production_l3937_393779


namespace imaginary_part_of_z_l3937_393780

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  let z := (4 : ℂ) / (1 - i)
  Complex.im z = 2 := by sorry

end imaginary_part_of_z_l3937_393780


namespace largest_constant_inequality_l3937_393747

theorem largest_constant_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (K : ℝ), K = Real.sqrt 3 ∧ 
  (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    Real.sqrt (x * y / z) + Real.sqrt (y * z / x) + Real.sqrt (x * z / y) ≥ K * Real.sqrt (x + y + z)) ∧
  (∀ (L : ℝ), 
    (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
      Real.sqrt (x * y / z) + Real.sqrt (y * z / x) + Real.sqrt (x * z / y) ≥ L * Real.sqrt (x + y + z)) →
    L ≤ K) :=
by sorry

end largest_constant_inequality_l3937_393747


namespace cost_of_dozen_pens_l3937_393768

/-- Given the cost of pens and pencils, prove the cost of one dozen pens -/
theorem cost_of_dozen_pens 
  (cost_3pens_5pencils : ℕ) 
  (ratio : ℚ) 
  (cost_dozen_pens : ℕ) 
  (h1 : cost_3pens_5pencils = 100)
  (h2 : ratio > 0)
  (h3 : cost_dozen_pens = 300) :
  cost_dozen_pens = 300 := by
  sorry

#check cost_of_dozen_pens

end cost_of_dozen_pens_l3937_393768


namespace workshop_workers_l3937_393724

theorem workshop_workers (average_salary : ℕ) (technician_count : ℕ) (technician_salary : ℕ) (non_technician_salary : ℕ) : 
  average_salary = 8000 →
  technician_count = 7 →
  technician_salary = 14000 →
  non_technician_salary = 6000 →
  ∃ (total_workers : ℕ), 
    total_workers * average_salary = technician_count * technician_salary + (total_workers - technician_count) * non_technician_salary ∧
    total_workers = 28 :=
by sorry

end workshop_workers_l3937_393724


namespace complex_inequality_l3937_393729

theorem complex_inequality (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
  Complex.abs (z - w) ≥ (1/2 : ℝ) * (Complex.abs z + Complex.abs w) * Complex.abs ((z / Complex.abs z) - (w / Complex.abs w)) ∧
  (Complex.abs (z - w) = (1/2 : ℝ) * (Complex.abs z + Complex.abs w) * Complex.abs ((z / Complex.abs z) - (w / Complex.abs w)) ↔
   (z / w).re < 0 ∨ Complex.abs z = Complex.abs w) :=
by sorry

end complex_inequality_l3937_393729


namespace second_die_sides_l3937_393703

theorem second_die_sides (n : ℕ) (h : n > 0) :
  (1 / 2) * ((n - 1) / (2 * n)) = 21428571428571427 / 100000000000000000 →
  n = 7 := by
sorry

end second_die_sides_l3937_393703


namespace numbers_with_seven_in_range_l3937_393717

/-- The count of natural numbers from 1 to 800 (inclusive) that contain the digit 7 at least once -/
def count_numbers_with_seven : ℕ := 152

/-- The total count of numbers from 1 to 800 -/
def total_count : ℕ := 800

/-- The count of numbers from 1 to 800 without the digit 7 -/
def count_without_seven : ℕ := 648

theorem numbers_with_seven_in_range :
  count_numbers_with_seven = total_count - count_without_seven :=
by sorry

end numbers_with_seven_in_range_l3937_393717


namespace tangent_circles_m_value_l3937_393714

/-- Two externally tangent circles C₁ and C₂ -/
structure TangentCircles where
  /-- Equation of C₁: (x+2)² + (y-m)² = 9 -/
  c1 : ∀ (x y : ℝ), (x + 2)^2 + (y - m)^2 = 9
  /-- Equation of C₂: (x-m)² + (y+1)² = 4 -/
  c2 : ∀ (x y : ℝ), (x - m)^2 + (y + 1)^2 = 4
  /-- m is a real number -/
  m : ℝ

/-- The value of m for externally tangent circles C₁ and C₂ is either 2 or -5 -/
theorem tangent_circles_m_value (tc : TangentCircles) : tc.m = 2 ∨ tc.m = -5 := by
  sorry

end tangent_circles_m_value_l3937_393714


namespace calculate_expression_l3937_393735

theorem calculate_expression : 4 + (-2)^2 * 2 + (-36) / 4 = 3 := by
  sorry

end calculate_expression_l3937_393735


namespace binomial_expansion_sum_l3937_393734

theorem binomial_expansion_sum (a₁ a₂ : ℕ) : 
  (∀ k : ℕ, k ≤ 10 → a₁ = 20 ∧ a₂ = 180) → 
  a₁ + a₂ = 200 := by
  sorry

end binomial_expansion_sum_l3937_393734


namespace negation_of_forall_implication_l3937_393790

theorem negation_of_forall_implication (A B : Set α) :
  (¬ (∀ x, x ∈ A → x ∈ B)) ↔ (∃ x, x ∈ A ∧ x ∉ B) := by
  sorry

end negation_of_forall_implication_l3937_393790


namespace circle_ratio_after_increase_l3937_393722

/-- The ratio of the new circumference to the new diameter when the radius is increased by 2 units -/
theorem circle_ratio_after_increase (r : ℝ) : 
  (2 * Real.pi * (r + 2)) / (2 * (r + 2)) = Real.pi :=
by sorry

end circle_ratio_after_increase_l3937_393722


namespace triangulated_square_interior_points_l3937_393706

/-- Represents a square divided into triangles -/
structure TriangulatedSquare where
  /-- The number of triangles in the square -/
  num_triangles : ℕ
  /-- The number of interior points (vertices of triangles) -/
  num_interior_points : ℕ
  /-- Condition: No vertex lies on sides or inside other triangles -/
  no_overlap : Prop
  /-- Condition: Sides of square are sides of some triangles -/
  square_sides_are_triangle_sides : Prop

/-- Theorem: A square divided into 2016 triangles has 1007 interior points -/
theorem triangulated_square_interior_points
  (ts : TriangulatedSquare)
  (h_num_triangles : ts.num_triangles = 2016) :
  ts.num_interior_points = 1007 := by
  sorry

#check triangulated_square_interior_points

end triangulated_square_interior_points_l3937_393706


namespace zero_of_composite_f_l3937_393776

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2 * Real.exp x else Real.log x

-- State the theorem
theorem zero_of_composite_f :
  ∃ (x : ℝ), f (f x) = 0 ∧ x = Real.exp 1 := by
  sorry

end zero_of_composite_f_l3937_393776


namespace interest_equality_implies_second_sum_l3937_393787

/-- Given a total sum split into two parts, if the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years
    at 5% per annum, then the second part is 1664 Rs. -/
theorem interest_equality_implies_second_sum (total : ℝ) (first second : ℝ) :
  total = 2704 →
  first + second = total →
  (first * 3 * 8) / 100 = (second * 5 * 3) / 100 →
  second = 1664 := by
  sorry

end interest_equality_implies_second_sum_l3937_393787


namespace nested_radicals_solution_l3937_393755

-- Define the left-hand side of the equation
noncomputable def leftSide (x : ℝ) : ℝ := 
  Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

-- Define the right-hand side of the equation
noncomputable def rightSide (x : ℝ) : ℝ := 
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

-- State the theorem
theorem nested_radicals_solution :
  ∃! (x : ℝ), x > 0 ∧ leftSide x = rightSide x :=
by
  -- The unique solution is 2
  use 2
  sorry

end nested_radicals_solution_l3937_393755


namespace walnut_count_l3937_393770

theorem walnut_count (total : ℕ) (p a c w : ℕ) : 
  total = 150 →
  p + a + c + w = total →
  a = p / 2 →
  c = 4 * a →
  w = 3 * c →
  w = 96 := by
  sorry

end walnut_count_l3937_393770


namespace joan_apple_picking_l3937_393774

/-- Given that Joan gave 27 apples to Melanie and now has 16 apples,
    prove that she picked 43 apples from the orchard. -/
theorem joan_apple_picking (apples_given : ℕ) (apples_left : ℕ) 
  (h1 : apples_given = 27) (h2 : apples_left = 16) :
  apples_given + apples_left = 43 := by
  sorry

end joan_apple_picking_l3937_393774


namespace egyptian_art_pieces_l3937_393746

theorem egyptian_art_pieces (total : ℕ) (asian : ℕ) (egyptian : ℕ) : 
  total = 992 → asian = 465 → egyptian = total - asian → egyptian = 527 := by
sorry

end egyptian_art_pieces_l3937_393746


namespace helga_wrote_250_articles_l3937_393708

/-- Represents Helga's work schedule and article production --/
structure HelgaWork where
  articles_per_30min : ℕ := 5
  usual_hours_per_day : ℕ := 4
  usual_days_per_week : ℕ := 5
  extra_hours_thursday : ℕ := 2
  extra_hours_friday : ℕ := 3

/-- Calculates the total number of articles Helga wrote in a week --/
def total_articles_in_week (h : HelgaWork) : ℕ :=
  let articles_per_hour := h.articles_per_30min * 2
  let usual_articles_per_day := articles_per_hour * h.usual_hours_per_day
  let usual_articles_per_week := usual_articles_per_day * h.usual_days_per_week
  let extra_articles_thursday := articles_per_hour * h.extra_hours_thursday
  let extra_articles_friday := articles_per_hour * h.extra_hours_friday
  usual_articles_per_week + extra_articles_thursday + extra_articles_friday

/-- Theorem stating that Helga wrote 250 articles in the given week --/
theorem helga_wrote_250_articles : 
  ∀ (h : HelgaWork), total_articles_in_week h = 250 := by
  sorry

end helga_wrote_250_articles_l3937_393708


namespace max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l3937_393792

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ

/-- The probability of a number in a segment being divisible by 10 -/
def probabilityDivisibleBy10 (s : Segment) : ℚ :=
  (s.length.div 10) / s.length

theorem max_probability_divisible_by_10 :
  ∃ s : Segment, probabilityDivisibleBy10 s = 1 ∧
  ∀ t : Segment, probabilityDivisibleBy10 t ≤ 1 :=
sorry

theorem min_nonzero_probability_divisible_by_10 :
  ∃ s : Segment, probabilityDivisibleBy10 s = 1/19 ∧
  ∀ t : Segment, probabilityDivisibleBy10 t = 0 ∨ probabilityDivisibleBy10 t ≥ 1/19 :=
sorry

end max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l3937_393792


namespace simplify_expression_l3937_393781

theorem simplify_expression (a : ℝ) : a + 1 + a - 2 + a + 3 + a - 4 = 4*a - 2 := by
  sorry

end simplify_expression_l3937_393781


namespace match_triangle_formation_l3937_393752

theorem match_triangle_formation (n : ℕ) : 
  (n = 100 → ¬(3 ∣ (n * (n + 1) / 2))) ∧ 
  (n = 99 → (3 ∣ (n * (n + 1) / 2))) := by
  sorry

end match_triangle_formation_l3937_393752


namespace largest_root_ratio_l3937_393733

-- Define the polynomials
def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4
def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

-- Define x₁ as the largest root of f
def x₁ : ℝ := sorry

-- Define x₂ as the largest root of g
def x₂ : ℝ := sorry

-- Theorem statement
theorem largest_root_ratio :
  x₂ / x₁ = 2 :=
sorry

end largest_root_ratio_l3937_393733


namespace perp_necessary_not_sufficient_l3937_393761

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "within" relation for a line in a plane
variable (within : Line → Plane → Prop)

theorem perp_necessary_not_sufficient
  (l m : Line) (α : Plane)
  (h_diff : l ≠ m)
  (h_within : within m α) :
  (∀ x : Line, within x α → perp_line_plane l α → perp_line_line l x) ∧
  (∃ β : Plane, perp_line_line l m ∧ ¬perp_line_plane l β ∧ within m β) :=
sorry

end perp_necessary_not_sufficient_l3937_393761


namespace intersection_of_sets_l3937_393767

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | x < 2}
  let B : Set ℝ := {x | 3 - 2*x > 0}
  A ∩ B = {x | x < 3/2} := by
sorry

end intersection_of_sets_l3937_393767


namespace cycle_gain_percent_l3937_393739

/-- The gain percent when a cycle is bought for 450 Rs and sold for 520 Rs -/
def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the gain percent is 15.56% -/
theorem cycle_gain_percent : 
  gain_percent 450 520 = 15.56 := by
  sorry

end cycle_gain_percent_l3937_393739


namespace percentage_loss_l3937_393718

theorem percentage_loss (cost_price selling_price : ℝ) : 
  cost_price = 750 → 
  selling_price = 675 → 
  (cost_price - selling_price) / cost_price * 100 = 10 := by
sorry

end percentage_loss_l3937_393718


namespace tim_final_coin_count_l3937_393730

/-- Represents the count of different types of coins -/
structure CoinCount where
  quarters : ℕ
  nickels : ℕ
  dimes : ℕ
  pennies : ℕ

/-- Represents a transaction that modifies the coin count -/
inductive Transaction
  | DadGift : Transaction
  | DadExchange : Transaction
  | PaySister : Transaction
  | BuySnack : Transaction
  | ExchangeQuarter : Transaction

def initial_coins : CoinCount :=
  { quarters := 7, nickels := 9, dimes := 12, pennies := 5 }

def apply_transaction (coins : CoinCount) (t : Transaction) : CoinCount :=
  match t with
  | Transaction.DadGift => 
      { quarters := coins.quarters + 2,
        nickels := coins.nickels + 3,
        dimes := coins.dimes,
        pennies := coins.pennies + 5 }
  | Transaction.DadExchange => 
      { quarters := coins.quarters + 4,
        nickels := coins.nickels,
        dimes := coins.dimes - 10,
        pennies := coins.pennies }
  | Transaction.PaySister => 
      { quarters := coins.quarters,
        nickels := coins.nickels - 5,
        dimes := coins.dimes,
        pennies := coins.pennies }
  | Transaction.BuySnack => 
      { quarters := coins.quarters - 2,
        nickels := coins.nickels - 4,
        dimes := coins.dimes,
        pennies := coins.pennies }
  | Transaction.ExchangeQuarter => 
      { quarters := coins.quarters - 1,
        nickels := coins.nickels + 5,
        dimes := coins.dimes,
        pennies := coins.pennies }

def final_coins : CoinCount :=
  apply_transaction
    (apply_transaction
      (apply_transaction
        (apply_transaction
          (apply_transaction initial_coins Transaction.DadGift)
          Transaction.DadExchange)
        Transaction.PaySister)
      Transaction.BuySnack)
    Transaction.ExchangeQuarter

theorem tim_final_coin_count :
  final_coins = { quarters := 10, nickels := 8, dimes := 2, pennies := 10 } :=
by sorry

end tim_final_coin_count_l3937_393730


namespace min_product_of_three_positive_reals_l3937_393773

theorem min_product_of_three_positive_reals (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 1/18 :=
by sorry

end min_product_of_three_positive_reals_l3937_393773


namespace problem_solution_l3937_393783

theorem problem_solution : (2010^2 - 2010 + 1) / (2010 + 1) = 4040091 / 2011 := by
  sorry

end problem_solution_l3937_393783


namespace average_customer_donation_l3937_393794

/-- Given a restaurant fundraiser where:
    1. The restaurant's donation is 1/5 of the total customer donation.
    2. There are 40 customers.
    3. The restaurant's total donation is $24.
    Prove that the average customer donation is $3. -/
theorem average_customer_donation (restaurant_ratio : ℚ) (num_customers : ℕ) (restaurant_donation : ℚ) :
  restaurant_ratio = 1 / 5 →
  num_customers = 40 →
  restaurant_donation = 24 →
  (restaurant_donation / restaurant_ratio) / num_customers = 3 := by
sorry

end average_customer_donation_l3937_393794


namespace parabola_vertex_l3937_393723

/-- The parabola defined by the equation y = (3x-1)^2 + 2 has vertex (1/3, 2) -/
theorem parabola_vertex (x y : ℝ) :
  y = (3*x - 1)^2 + 2 →
  (∃ a h k : ℝ, a ≠ 0 ∧ y = a*(x - h)^2 + k ∧ h = 1/3 ∧ k = 2) :=
by sorry

end parabola_vertex_l3937_393723


namespace max_dominoes_after_removal_l3937_393753

/-- Represents a chessboard with some squares removed -/
structure Chessboard :=
  (size : Nat)
  (removed : Nat)
  (removed_black : Nat)
  (removed_white : Nat)

/-- Calculates the maximum number of guaranteed dominoes -/
def max_guaranteed_dominoes (board : Chessboard) : Nat :=
  sorry

/-- Theorem stating the maximum number of guaranteed dominoes for the given problem -/
theorem max_dominoes_after_removal :
  ∀ (board : Chessboard),
    board.size = 8 ∧
    board.removed = 10 ∧
    board.removed_black > 0 ∧
    board.removed_white > 0 ∧
    board.removed_black + board.removed_white = board.removed →
    max_guaranteed_dominoes board = 23 :=
  sorry

end max_dominoes_after_removal_l3937_393753


namespace max_value_complex_expression_l3937_393797

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs (z + Complex.I) = 2) :
  ∃ (max_val : ℝ), max_val = 4 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs (w + Complex.I) = 2 →
    Complex.abs ((w - (2 - Complex.I))^2 * (w - Complex.I)) ≤ max_val :=
sorry

end max_value_complex_expression_l3937_393797


namespace complement_of_angle_l3937_393712

theorem complement_of_angle (A : ℝ) (h : A = 35) : 90 - A = 55 := by
  sorry

end complement_of_angle_l3937_393712


namespace quadratic_form_sum_l3937_393795

theorem quadratic_form_sum (x : ℝ) : ∃ (b c : ℝ), 
  (x^2 - 26*x + 81 = (x + b)^2 + c) ∧ (b + c = -101) := by
  sorry

end quadratic_form_sum_l3937_393795


namespace f_negative_two_eq_one_l3937_393788

/-- The function f(x) defined as x^5 + ax^3 + x^2 + bx + 2 -/
noncomputable def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + x^2 + b*x + 2

/-- Theorem: If f(2) = 3, then f(-2) = 1 -/
theorem f_negative_two_eq_one (a b : ℝ) (h : f a b 2 = 3) : f a b (-2) = 1 := by
  sorry

end f_negative_two_eq_one_l3937_393788


namespace min_sales_to_break_even_l3937_393737

def current_salary : ℕ := 90000
def new_base_salary : ℕ := 45000
def sale_value : ℕ := 1500
def commission_rate : ℚ := 15 / 100

theorem min_sales_to_break_even : 
  ∃ (n : ℕ), n = 200 ∧ 
  (n : ℚ) * commission_rate * sale_value + new_base_salary = current_salary ∧
  ∀ (m : ℕ), m < n → (m : ℚ) * commission_rate * sale_value + new_base_salary < current_salary :=
sorry

end min_sales_to_break_even_l3937_393737


namespace cosine_value_from_sine_l3937_393736

theorem cosine_value_from_sine (θ : Real) (h1 : 0 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin (θ / 2 + π / 6) = 3 / 5) : 
  Real.cos (θ + 5 * π / 6) = -24 / 25 := by
  sorry

end cosine_value_from_sine_l3937_393736


namespace quadratic_equations_solutions_l3937_393791

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 7 + 2 * Real.sqrt 7 ∧ x₂ = 7 - 2 * Real.sqrt 7 ∧
    x₁^2 - 14*x₁ + 21 = 0 ∧ x₂^2 - 14*x₂ + 21 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2 ∧
    x₁^2 - 3*x₁ + 2 = 0 ∧ x₂^2 - 3*x₂ + 2 = 0) :=
by sorry

end quadratic_equations_solutions_l3937_393791


namespace locus_of_D_l3937_393707

-- Define the basic structure for points in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a function to calculate the area of a triangle
def triangleArea (A B C : Point) : ℝ := sorry

-- Define a function to calculate the area of a quadrilateral
def quadArea (A B C D : Point) : ℝ := sorry

-- Define a function to check if three points are collinear
def collinear (A B C : Point) : Prop := sorry

-- Define a function to calculate the distance from a point to a line
def distanceToLine (P : Point) (A B : Point) : ℝ := sorry

-- Define a function to check if a point is on a line
def onLine (P : Point) (A B : Point) : Prop := sorry

-- Theorem statement
theorem locus_of_D (A B C D : Point) :
  ¬collinear A B C →
  quadArea A B C D = 3 * triangleArea A B C →
  ∃ (k : ℝ), distanceToLine D A C = 4 * distanceToLine B A C ∧
             ¬onLine D A B ∧
             ¬onLine D B C :=
sorry

end locus_of_D_l3937_393707


namespace necessary_and_sufficient_condition_l3937_393760

theorem necessary_and_sufficient_condition (a b : ℝ) : (a > b) ↔ (a * |a| > b * |b|) := by
  sorry

end necessary_and_sufficient_condition_l3937_393760


namespace balloon_difference_l3937_393732

theorem balloon_difference (x y z : ℚ) 
  (eq1 : x = 3 * z - 2)
  (eq2 : y = z / 4 + 5)
  (eq3 : z = y + 3) :
  x + y - z = 27 := by
  sorry

end balloon_difference_l3937_393732


namespace smallest_number_divisible_by_twelve_after_subtracting_seven_l3937_393727

theorem smallest_number_divisible_by_twelve_after_subtracting_seven : 
  ∃ N : ℕ, N > 0 ∧ (N - 7) % 12 = 0 ∧ ∀ m : ℕ, m > 0 → (m - 7) % 12 = 0 → m ≥ N := by
  sorry

end smallest_number_divisible_by_twelve_after_subtracting_seven_l3937_393727


namespace heavens_brother_erasers_l3937_393757

def total_money : ℕ := 100
def sharpener_count : ℕ := 2
def notebook_count : ℕ := 4
def item_price : ℕ := 5
def eraser_price : ℕ := 4
def highlighter_cost : ℕ := 30

theorem heavens_brother_erasers :
  let heaven_spent := sharpener_count * item_price + notebook_count * item_price
  let brother_money := total_money - heaven_spent
  let eraser_money := brother_money - highlighter_cost
  eraser_money / eraser_price = 10 := by sorry

end heavens_brother_erasers_l3937_393757


namespace chess_team_boys_count_l3937_393700

theorem chess_team_boys_count (total_members : ℕ) (meeting_attendees : ℕ) : 
  total_members = 30 →
  meeting_attendees = 20 →
  ∃ (girls : ℕ) (boys : ℕ),
    girls + boys = total_members ∧
    (2 * girls / 3 : ℚ) + boys = meeting_attendees ∧
    boys = 0 := by
  sorry

end chess_team_boys_count_l3937_393700


namespace quadratic_roots_relation_l3937_393741

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ r₁ r₂ : ℝ, r₁ + r₂ = -p ∧ r₁ * r₂ = m ∧
    r₁ / 2 + r₂ / 2 = -m ∧ (r₁ / 2) * (r₂ / 2) = n) →
  n / p = 1 / 8 := by
sorry

end quadratic_roots_relation_l3937_393741


namespace brick_length_l3937_393796

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The length of a brick with given dimensions and surface area -/
theorem brick_length (w h SA : ℝ) (hw : w = 4) (hh : h = 2) (hSA : SA = 112) :
  ∃ l : ℝ, surface_area l w h = SA ∧ l = 8 := by
  sorry

end brick_length_l3937_393796


namespace dawn_time_verify_solution_l3937_393756

/-- Represents the time in hours from dawn to noon -/
def time_dawn_to_noon : ℝ := sorry

/-- Represents the time in hours from noon to 4 PM -/
def time_noon_to_4pm : ℝ := 4

/-- Represents the time in hours from noon to 9 PM -/
def time_noon_to_9pm : ℝ := 9

/-- The theorem stating that the time from dawn to noon is 6 hours -/
theorem dawn_time : time_dawn_to_noon = 6 := by
  sorry

/-- Verification of the solution using speed ratios -/
theorem verify_solution :
  time_dawn_to_noon / time_noon_to_4pm = time_noon_to_9pm / time_dawn_to_noon := by
  sorry

end dawn_time_verify_solution_l3937_393756


namespace tenth_term_of_sequence_l3937_393745

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence :
  let a₁ := (3 : ℚ) / 4
  let a₂ := (5 : ℚ) / 4
  let a₃ := (7 : ℚ) / 4
  let d := a₂ - a₁
  arithmeticSequence a₁ d 10 = (21 : ℚ) / 4 := by
  sorry

end tenth_term_of_sequence_l3937_393745


namespace complex_equation_solution_l3937_393754

theorem complex_equation_solution (x y : ℝ) : 
  (2 * x - y - 2 : ℂ) + (y - 2 : ℂ) * I = 0 → x = 2 ∧ y = 2 := by
  sorry

end complex_equation_solution_l3937_393754


namespace small_hotdogs_count_l3937_393759

theorem small_hotdogs_count (total : ℕ) (large : ℕ) (h1 : total = 79) (h2 : large = 21) :
  total - large = 58 := by
  sorry

end small_hotdogs_count_l3937_393759


namespace total_savings_l3937_393713

/-- Represents the savings of Anne and Katherine -/
structure Savings where
  anne : ℝ
  katherine : ℝ

/-- The conditions of the savings problem -/
def SavingsConditions (s : Savings) : Prop :=
  (s.anne - 150 = (1 / 3) * s.katherine) ∧
  (2 * s.katherine = 3 * s.anne)

/-- Theorem stating that under the given conditions, the total savings is $750 -/
theorem total_savings (s : Savings) (h : SavingsConditions s) : 
  s.anne + s.katherine = 750 := by
  sorry

#check total_savings

end total_savings_l3937_393713


namespace inequality_proof_l3937_393769

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) 
  (h5 : a + b + c + d = 8) : 
  (a^3 / (a^2 + b + c)) + (b^3 / (b^2 + c + d)) + 
  (c^3 / (c^2 + d + a)) + (d^3 / (d^2 + a + b)) ≥ 4 := by
  sorry

end inequality_proof_l3937_393769


namespace painting_selection_theorem_l3937_393702

/-- Number of traditional Chinese paintings -/
def traditional_paintings : ℕ := 5

/-- Number of oil paintings -/
def oil_paintings : ℕ := 2

/-- Number of watercolor paintings -/
def watercolor_paintings : ℕ := 7

/-- Number of ways to choose one painting from each category -/
def one_from_each : ℕ := traditional_paintings * oil_paintings * watercolor_paintings

/-- Number of ways to choose two paintings of different types -/
def two_different_types : ℕ := 
  traditional_paintings * oil_paintings + 
  traditional_paintings * watercolor_paintings + 
  oil_paintings * watercolor_paintings

theorem painting_selection_theorem : 
  one_from_each = 70 ∧ two_different_types = 59 := by sorry

end painting_selection_theorem_l3937_393702


namespace sibling_pair_probability_l3937_393764

theorem sibling_pair_probability (business_students : ℕ) (law_students : ℕ) (sibling_pairs : ℕ) : 
  business_students = 500 →
  law_students = 800 →
  sibling_pairs = 30 →
  (sibling_pairs : ℚ) / (business_students * law_students) = 0.000075 := by
  sorry

end sibling_pair_probability_l3937_393764


namespace scheme2_more_cost_effective_l3937_393771

/-- Represents the payment for Scheme 1 -/
def scheme1_payment (x : ℕ) : ℚ :=
  90 * (1 - 30/100) * x + 100 * (1 - 15/100) * (2*x + 1)

/-- Represents the payment for Scheme 2 -/
def scheme2_payment (x : ℕ) : ℚ :=
  (90*x + 100*(2*x + 1)) * (1 - 20/100)

/-- Theorem stating that Scheme 2 is more cost-effective for x ≥ 33 -/
theorem scheme2_more_cost_effective (x : ℕ) (h : x ≥ 33) :
  scheme2_payment x < scheme1_payment x :=
sorry

end scheme2_more_cost_effective_l3937_393771


namespace venerts_in_45_degrees_l3937_393786

/-- Converts degrees to venerts given the number of venerts in a full circle -/
def degrees_to_venerts (full_circle_venerts : ℚ) (degrees : ℚ) : ℚ :=
  (degrees * full_circle_venerts) / 360

/-- Theorem: Given 600 venerts in a full circle, 45° is equivalent to 75 venerts -/
theorem venerts_in_45_degrees :
  degrees_to_venerts 600 45 = 75 := by
  sorry

end venerts_in_45_degrees_l3937_393786


namespace termite_ridden_not_collapsing_l3937_393758

theorem termite_ridden_not_collapsing (total_homes : ℚ) 
  (termite_ridden_ratio : ℚ) (collapsing_ratio : ℚ) :
  termite_ridden_ratio = 1/3 →
  collapsing_ratio = 5/8 →
  termite_ridden_ratio - (termite_ridden_ratio * collapsing_ratio) = 1/8 :=
by sorry

end termite_ridden_not_collapsing_l3937_393758


namespace smallest_portion_l3937_393738

def bread_distribution (a : ℚ) (d : ℚ) : Prop :=
  -- Total sum is 100
  5 * a + 10 * d = 100 ∧
  -- Sum of largest three portions is 1/7 of sum of smaller two
  (3 * a + 6 * d) = (1/7) * (2 * a + d)

theorem smallest_portion : 
  ∃ (a d : ℚ), bread_distribution a d ∧ a = 5/3 :=
sorry

end smallest_portion_l3937_393738


namespace concatenation_problem_l3937_393775

theorem concatenation_problem :
  ∃ (a b : ℕ),
    100 ≤ a ∧ a ≤ 999 ∧
    1000 ≤ b ∧ b ≤ 9999 ∧
    10000 * a + b = 11 * a * b ∧
    a + b = 1093 := by
  sorry

end concatenation_problem_l3937_393775


namespace hot_dog_contest_l3937_393720

/-- Hot dog eating contest problem -/
theorem hot_dog_contest (first_competitor second_competitor third_competitor : ℕ) : 
  first_competitor = 12 →
  second_competitor = 2 * first_competitor →
  third_competitor = second_competitor - (second_competitor / 4) →
  third_competitor = 18 := by
  sorry

end hot_dog_contest_l3937_393720


namespace chess_tournament_games_l3937_393785

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 20 players, where each player plays every other player
    exactly once, and each game involves two players, the total number of games played is 190. --/
theorem chess_tournament_games :
  num_games 20 = 190 := by
  sorry

end chess_tournament_games_l3937_393785


namespace arrange_six_books_three_identical_l3937_393704

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem: Arranging 6 books with 3 identical copies results in 120 ways -/
theorem arrange_six_books_three_identical :
  arrange_books 6 3 = 120 := by
  sorry

end arrange_six_books_three_identical_l3937_393704


namespace sum_reciprocal_lower_bound_l3937_393709

theorem sum_reciprocal_lower_bound (a₁ a₂ a₃ : ℝ) 
  (h_pos₁ : a₁ > 0) (h_pos₂ : a₂ > 0) (h_pos₃ : a₃ > 0)
  (h_sum : a₁ + a₂ + a₃ = 1) : 
  1/a₁ + 1/a₂ + 1/a₃ ≥ 9 := by
sorry

end sum_reciprocal_lower_bound_l3937_393709


namespace closest_multiple_of_15_to_500_l3937_393762

def multiple_of_15 (n : ℤ) : ℤ := 15 * n

theorem closest_multiple_of_15_to_500 :
  ∀ k : ℤ, k ≠ 33 → |500 - multiple_of_15 33| ≤ |500 - multiple_of_15 k| :=
by sorry

end closest_multiple_of_15_to_500_l3937_393762


namespace balloon_ratio_l3937_393766

theorem balloon_ratio (sally_balloons fred_balloons : ℕ) 
  (h1 : sally_balloons = 6) 
  (h2 : fred_balloons = 18) : 
  (fred_balloons : ℚ) / sally_balloons = 3 := by
  sorry

end balloon_ratio_l3937_393766


namespace expected_defective_theorem_l3937_393721

/-- The expected number of defective products drawn before a genuine product is drawn -/
def expected_defective_drawn (genuine : ℕ) (defective : ℕ) : ℚ :=
  -- Definition to be filled based on the problem conditions
  sorry

theorem expected_defective_theorem :
  expected_defective_drawn 9 3 = 9/5 := by
  sorry

end expected_defective_theorem_l3937_393721


namespace probability_is_half_l3937_393778

/-- Represents a game board as described in the problem -/
structure GameBoard where
  total_regions : ℕ
  shaded_regions : ℕ
  h_total : total_regions = 8
  h_shaded : shaded_regions = 4

/-- The probability of landing in a shaded region on the game board -/
def probability (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- Theorem stating that the probability of landing in a shaded region is 1/2 -/
theorem probability_is_half (board : GameBoard) : probability board = 1/2 := by
  sorry

end probability_is_half_l3937_393778


namespace cafeteria_fruit_distribution_l3937_393710

/-- The number of students who wanted fruit in the school cafeteria -/
def students_wanting_fruit : ℕ := 21

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 6

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 15

/-- The number of extra apples left after distribution -/
def extra_apples : ℕ := 16

/-- Theorem stating that the number of students who wanted fruit is 21 -/
theorem cafeteria_fruit_distribution :
  students_wanting_fruit = red_apples + green_apples :=
by
  sorry

#check cafeteria_fruit_distribution

end cafeteria_fruit_distribution_l3937_393710


namespace linearDependence_l3937_393716

/-- Two 2D vectors -/
def v1 : Fin 2 → ℝ := ![2, 4]
def v2 (k : ℝ) : Fin 2 → ℝ := ![4, k]

/-- The set of vectors is linearly dependent iff there exist non-zero scalars a and b
    such that a * v1 + b * v2 = 0 -/
def isLinearlyDependent (k : ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (∀ i, a * v1 i + b * v2 k i = 0)

theorem linearDependence (k : ℝ) : isLinearlyDependent k ↔ k = 8 := by
  sorry

end linearDependence_l3937_393716


namespace polynomial_remainder_l3937_393799

def f (x : ℝ) : ℝ := 5*x^6 - 3*x^4 + 6*x^3 - 8*x + 10

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, f = λ x => (3*x - 9) * q x + 3550 :=
sorry

end polynomial_remainder_l3937_393799


namespace gcd_multiple_smallest_l3937_393748

/-- Given positive integers m and n with gcd(m,n) = 12, 
    the smallest possible value of gcd(12m,18n) is 72 -/
theorem gcd_multiple_smallest (m n : ℕ+) (h : Nat.gcd m n = 12) :
  ∃ (k : ℕ+), ∀ (x : ℕ+), Nat.gcd (12 * m) (18 * n) ≥ k ∧ 
  (∃ (m' n' : ℕ+), Nat.gcd m' n' = 12 ∧ Nat.gcd (12 * m') (18 * n') = k) ∧
  k = 72 := by
  sorry

#check gcd_multiple_smallest

end gcd_multiple_smallest_l3937_393748


namespace test_problems_count_l3937_393731

theorem test_problems_count :
  let total_points : ℕ := 110
  let computation_problems : ℕ := 20
  let points_per_computation : ℕ := 3
  let points_per_word : ℕ := 5
  let word_problems : ℕ := (total_points - computation_problems * points_per_computation) / points_per_word
  computation_problems + word_problems = 30 := by
  sorry

end test_problems_count_l3937_393731


namespace shark_teeth_relationship_hammerhead_shark_teeth_fraction_l3937_393711

/-- The number of teeth a tiger shark has -/
def tiger_shark_teeth : ℕ := 180

/-- The number of teeth a great white shark has -/
def great_white_shark_teeth : ℕ := 420

/-- The fraction of teeth a hammerhead shark has compared to a tiger shark -/
def hammerhead_fraction : ℚ := 1 / 6

/-- Theorem stating the relationship between shark teeth counts -/
theorem shark_teeth_relationship : 
  great_white_shark_teeth = 2 * (tiger_shark_teeth + hammerhead_fraction * tiger_shark_teeth) :=
by sorry

/-- Theorem proving the fraction of teeth a hammerhead shark has compared to a tiger shark -/
theorem hammerhead_shark_teeth_fraction : 
  hammerhead_fraction = 1 / 6 :=
by sorry

end shark_teeth_relationship_hammerhead_shark_teeth_fraction_l3937_393711
