import Mathlib

namespace four_solutions_to_simultaneous_equations_l1822_182290

theorem four_solutions_to_simultaneous_equations :
  ∃! (s : Finset (ℝ × ℝ)), (∀ (p : ℝ × ℝ), p ∈ s ↔ p.1^2 - p.2 = 2022 ∧ p.2^2 - p.1 = 2022) ∧ s.card = 4 := by
  sorry

end four_solutions_to_simultaneous_equations_l1822_182290


namespace parallel_transitivity_l1822_182204

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "outside of plane" relation
variable (outside_of_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity 
  (m n : Line) (α : Plane) 
  (h1 : outside_of_plane m α)
  (h2 : outside_of_plane n α)
  (h3 : parallel_lines m n)
  (h4 : parallel_line_plane m α) :
  parallel_line_plane n α :=
sorry

end parallel_transitivity_l1822_182204


namespace book_pages_l1822_182220

/-- The number of days Lex read the book -/
def days : ℕ := 12

/-- The number of pages Lex read per day -/
def pages_per_day : ℕ := 20

/-- The total number of pages in the book -/
def total_pages : ℕ := days * pages_per_day

theorem book_pages : total_pages = 240 := by sorry

end book_pages_l1822_182220


namespace line_parallel_to_parallel_plane_l1822_182208

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallelPlanes : Plane → Plane → Prop)
variable (perpendicularLinePlane : Line → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (m n : Line) (α β : Plane)
  (h1 : parallelPlanes α β)
  (h2 : perpendicularLinePlane m α)
  (h3 : perpendicularLinePlane n β) :
  parallelLinePlane m β :=
sorry

end line_parallel_to_parallel_plane_l1822_182208


namespace bead_system_eventually_repeats_l1822_182242

-- Define the bead system
structure BeadSystem where
  n : ℕ  -- number of beads
  ω : ℝ  -- angular speed
  direction : Fin n → Bool  -- true for clockwise, false for counterclockwise
  initial_position : Fin n → ℝ  -- initial angular position of each bead

-- Define the state of the system at a given time
def system_state (bs : BeadSystem) (t : ℝ) : Fin bs.n → ℝ :=
  sorry

-- Define what it means for the system to repeat its initial configuration
def repeats_initial_config (bs : BeadSystem) (t : ℝ) : Prop :=
  ∃ (perm : Equiv.Perm (Fin bs.n)),
    ∀ i, system_state bs t (perm i) = bs.initial_position i

-- State the theorem
theorem bead_system_eventually_repeats (bs : BeadSystem) :
  ∃ t > 0, repeats_initial_config bs t :=
sorry

end bead_system_eventually_repeats_l1822_182242


namespace sqrt_2x_minus_4_meaningful_l1822_182217

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 :=
by sorry

end sqrt_2x_minus_4_meaningful_l1822_182217


namespace discount_percentage_l1822_182230

theorem discount_percentage (original_price sale_price : ℝ) 
  (h1 : original_price = 600)
  (h2 : sale_price = 480) : 
  (original_price - sale_price) / original_price * 100 = 20 := by
  sorry

end discount_percentage_l1822_182230


namespace range_of_a_l1822_182261

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a-1)*x + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

end range_of_a_l1822_182261


namespace complex_product_l1822_182269

theorem complex_product (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2) 
  (h2 : Complex.abs z₂ = 3) 
  (h3 : 3 * z₁ - 2 * z₂ = 2 - I) : 
  z₁ * z₂ = -30/13 + 72/13 * I := by
sorry

end complex_product_l1822_182269


namespace dasha_number_l1822_182245

-- Define a function to calculate the product of digits
def digitProduct (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digitProduct (n / 10)

-- Define a function to check if a number is single-digit
def isSingleDigit (n : ℕ) : Prop := n < 10

-- Theorem statement
theorem dasha_number (n : ℕ) :
  n ≤ digitProduct n → isSingleDigit n :=
by sorry

end dasha_number_l1822_182245


namespace banana_permutations_l1822_182252

-- Define the word and its properties
def word : String := "BANANA"
def word_length : Nat := 6
def b_count : Nat := 1
def a_count : Nat := 3
def n_count : Nat := 2

-- Theorem statement
theorem banana_permutations :
  (Nat.factorial word_length) / 
  (Nat.factorial b_count * Nat.factorial a_count * Nat.factorial n_count) = 60 := by
  sorry

end banana_permutations_l1822_182252


namespace total_unique_customers_l1822_182231

/-- Represents the number of customers who had meals with both ham and cheese -/
def ham_cheese : ℕ := 80

/-- Represents the number of customers who had meals with both ham and tomatoes -/
def ham_tomato : ℕ := 90

/-- Represents the number of customers who had meals with both tomatoes and cheese -/
def tomato_cheese : ℕ := 100

/-- Represents the number of customers who had meals with all three ingredients -/
def all_three : ℕ := 20

/-- Theorem stating that the total number of unique customers is 230 -/
theorem total_unique_customers : 
  ham_cheese + ham_tomato + tomato_cheese - 2 * all_three = 230 := by
  sorry

end total_unique_customers_l1822_182231


namespace equal_angles_implies_rectangle_l1822_182292

-- Define a quadrilateral
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define the concept of equal angles in a quadrilateral
def has_four_equal_angles (q : Quadrilateral) : Prop := sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem equal_angles_implies_rectangle (q : Quadrilateral) :
  has_four_equal_angles q → is_rectangle q := by sorry

end equal_angles_implies_rectangle_l1822_182292


namespace tan_pi_plus_alpha_problem_l1822_182255

theorem tan_pi_plus_alpha_problem (α : Real) (h : Real.tan (Real.pi + α) = -1/2) :
  (2 * Real.cos (Real.pi - α) - 3 * Real.sin (Real.pi + α)) /
  (4 * Real.cos (α - 2 * Real.pi) + Real.cos (3/2 * Real.pi - α)) = -7/9 ∧
  Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 21/5 := by
  sorry

end tan_pi_plus_alpha_problem_l1822_182255


namespace logarithm_sum_equals_two_l1822_182266

theorem logarithm_sum_equals_two : Real.log 25 / Real.log 10 + (Real.log 2 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) = 2 := by
  sorry

end logarithm_sum_equals_two_l1822_182266


namespace sqrt_product_equals_thirty_l1822_182286

theorem sqrt_product_equals_thirty (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (5 * x) * Real.sqrt (30 * x) = 30) : 
  x = 1 / Real.sqrt 6 := by
sorry

end sqrt_product_equals_thirty_l1822_182286


namespace solution_set_l1822_182267

theorem solution_set (x : ℝ) : 4 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ x ∈ Set.Ioc (5/2) (20/7) :=
  sorry

end solution_set_l1822_182267


namespace correct_division_l1822_182221

theorem correct_division (n : ℕ) : 
  n % 8 = 2 ∧ n / 8 = 156 → n / 5 = 250 := by
  sorry

end correct_division_l1822_182221


namespace paintbrush_cost_l1822_182285

theorem paintbrush_cost (paint_cost easel_cost albert_has albert_needs : ℚ) :
  paint_cost = 4.35 →
  easel_cost = 12.65 →
  albert_has = 6.50 →
  albert_needs = 12 →
  paint_cost + easel_cost + (albert_has + albert_needs - (paint_cost + easel_cost)) = 1.50 := by
sorry

end paintbrush_cost_l1822_182285


namespace larger_box_jellybeans_l1822_182232

def jellybeans_in_box (length width height : ℕ) : ℕ := length * width * height * 20

theorem larger_box_jellybeans (l w h : ℕ) :
  jellybeans_in_box l w h = 200 →
  jellybeans_in_box (3 * l) (3 * w) (3 * h) = 5400 :=
by
  sorry

#check larger_box_jellybeans

end larger_box_jellybeans_l1822_182232


namespace whitewashing_cost_is_6770_l1822_182212

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ)
                     (doorLength doorWidth : ℝ)
                     (windowLength windowWidth : ℝ)
                     (numDoors numWindows : ℕ)
                     (costPerSqFt : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := numDoors * (doorLength * doorWidth)
  let windowArea := numWindows * (windowLength * windowWidth)
  let paintableArea := wallArea - doorArea - windowArea
  paintableArea * costPerSqFt

/-- Theorem stating that the cost of white washing the room with given specifications is 6770 Rs. -/
theorem whitewashing_cost_is_6770 :
  whitewashingCost 30 20 15 7 4 5 3 2 6 5 = 6770 := by
  sorry

end whitewashing_cost_is_6770_l1822_182212


namespace equal_roots_quadratic_l1822_182299

/-- A quadratic equation x^2 + 2x - c = 0 has two equal real roots if and only if c = -1 -/
theorem equal_roots_quadratic (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - c = 0 ∧ (∀ y : ℝ, y^2 + 2*y - c = 0 → y = x)) ↔ c = -1 :=
by sorry

end equal_roots_quadratic_l1822_182299


namespace average_score_calculation_l1822_182289

theorem average_score_calculation (total_students : ℝ) (male_ratio : ℝ) 
  (male_avg_score : ℝ) (female_avg_score : ℝ) 
  (h1 : male_ratio = 0.4)
  (h2 : male_avg_score = 75)
  (h3 : female_avg_score = 80) :
  (male_ratio * male_avg_score + (1 - male_ratio) * female_avg_score) = 78 := by
  sorry

#check average_score_calculation

end average_score_calculation_l1822_182289


namespace diana_operations_l1822_182278

theorem diana_operations (x : ℝ) : 
  (((x + 3) * 3 - 3) / 3 + 3 = 12) → x = 7 := by
sorry

end diana_operations_l1822_182278


namespace fruit_condition_percentage_l1822_182254

/-- Calculates the percentage of fruits in good condition given the number of oranges and bananas and their rotten percentages -/
def percentageGoodFruits (totalOranges totalBananas : ℕ) (rottenOrangesPercent rottenBananasPercent : ℚ) : ℚ :=
  let goodOranges : ℚ := totalOranges * (1 - rottenOrangesPercent)
  let goodBananas : ℚ := totalBananas * (1 - rottenBananasPercent)
  let totalFruits : ℚ := totalOranges + totalBananas
  (goodOranges + goodBananas) / totalFruits * 100

/-- Theorem stating that given 600 oranges with 15% rotten and 400 bananas with 4% rotten, 
    the percentage of fruits in good condition is 89.4% -/
theorem fruit_condition_percentage : 
  percentageGoodFruits 600 400 (15/100) (4/100) = 89.4 := by
  sorry

end fruit_condition_percentage_l1822_182254


namespace distinct_positive_numbers_properties_l1822_182297

theorem distinct_positive_numbers_properties (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 > 0) ∧ 
  (a > b ∨ a < b ∨ a = b) ∧
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) :=
by sorry

end distinct_positive_numbers_properties_l1822_182297


namespace officer_selection_l1822_182283

theorem officer_selection (n m k l : ℕ) (hn : n = 20) (hm : m = 8) (hk : k = 10) (hl : l = 3) :
  Nat.choose n m - (Nat.choose k m + Nat.choose k 1 * Nat.choose (n - k) (m - 1) + Nat.choose k 2 * Nat.choose (n - k) (m - 2)) = 115275 :=
by sorry

end officer_selection_l1822_182283


namespace tan_240_degrees_l1822_182202

theorem tan_240_degrees : Real.tan (240 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

#check tan_240_degrees

end tan_240_degrees_l1822_182202


namespace therapy_charges_relation_l1822_182293

/-- A psychologist's charging scheme for therapy sessions. -/
structure TherapyCharges where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  first_hour_premium : firstHourCharge = additionalHourCharge + 30

/-- Calculate the total charge for a given number of therapy hours. -/
def totalCharge (charges : TherapyCharges) (hours : ℕ) : ℕ :=
  charges.firstHourCharge + (hours - 1) * charges.additionalHourCharge

/-- Theorem stating the relationship between charges for 5 hours and 3 hours of therapy. -/
theorem therapy_charges_relation (charges : TherapyCharges) :
  totalCharge charges 5 = 400 → totalCharge charges 3 = 252 := by
  sorry

#check therapy_charges_relation

end therapy_charges_relation_l1822_182293


namespace no_solution_implies_a_le_8_l1822_182282

theorem no_solution_implies_a_le_8 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end no_solution_implies_a_le_8_l1822_182282


namespace newspaper_selling_price_l1822_182215

theorem newspaper_selling_price 
  (total_newspapers : ℕ) 
  (sold_percentage : ℚ)
  (buying_discount : ℚ)
  (total_profit : ℚ) :
  total_newspapers = 500 →
  sold_percentage = 80 / 100 →
  buying_discount = 75 / 100 →
  total_profit = 550 →
  ∃ (selling_price : ℚ),
    selling_price = 2 ∧
    (sold_percentage * total_newspapers : ℚ) * selling_price -
    (1 - buying_discount) * (total_newspapers : ℚ) * selling_price = total_profit :=
by sorry

end newspaper_selling_price_l1822_182215


namespace geometric_sequence_property_l1822_182223

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    for all n, a(n+1) = r * a(n) -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometric a) 
    (h1 : a 1 * a 99 = 16) : a 20 * a 80 = 16 := by
  sorry

end geometric_sequence_property_l1822_182223


namespace tan_equality_integer_l1822_182205

theorem tan_equality_integer (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → n = -30 := by
  sorry

end tan_equality_integer_l1822_182205


namespace circular_path_area_l1822_182234

/-- The area of a circular path around a circular lawn -/
theorem circular_path_area (r : ℝ) (w : ℝ) (h_r : r > 0) (h_w : w > 0) :
  let R := r + w
  (π * R^2 - π * r^2) = π * (R^2 - r^2) := by sorry

#check circular_path_area

end circular_path_area_l1822_182234


namespace ab_value_l1822_182281

theorem ab_value (a b : ℕ+) (h1 : a + b = 30) (h2 : 3 * a * b + 5 * a = 4 * b + 180) : a * b = 29 := by
  sorry

end ab_value_l1822_182281


namespace complex_fraction_equality_l1822_182201

theorem complex_fraction_equality : (2 : ℂ) / (1 - Complex.I) = 1 + Complex.I := by
  sorry

end complex_fraction_equality_l1822_182201


namespace quadratic_max_min_l1822_182253

def f (x : ℝ) := x^2 - 4*x + 2

theorem quadratic_max_min :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 5, f x ≤ max ∧ min ≤ f x) ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) 5, f x₁ = max) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) 5, f x₂ = min) ∧
    max = 14 ∧ min = -2 :=
by sorry

end quadratic_max_min_l1822_182253


namespace hyperbola_equation_l1822_182225

/-- Given a hyperbola with eccentricity 2 and foci coinciding with those of a specific ellipse,
    prove that its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h : ℝ × ℝ → Prop) (e : ℝ × ℝ → Prop) :
  (∀ x y, h (x, y) ↔ x^2/a^2 - y^2/b^2 = 1) →
  (∀ x y, e (x, y) ↔ x^2/25 + y^2/9 = 1) →
  (∀ x y, h (x, y) → (x/a)^2 - (y/b)^2 = 1) →
  (∀ x, e (x, 0) → x = 4 ∨ x = -4) →
  (∀ x, h (x, 0) → x = 4 ∨ x = -4) →
  (a / Real.sqrt (a^2 - b^2) = 2) →
  (∀ x y, h (x, y) ↔ x^2/4 - y^2/12 = 1) :=
by sorry

end hyperbola_equation_l1822_182225


namespace vector_magnitude_problem_l1822_182259

/-- Given two vectors a and b in ℝ², where b = (-1, 2) and a + b = (1, 3),
    prove that the magnitude of a - 2b is 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  b = (-1, 2) → a + b = (1, 3) → ‖a - 2 • b‖ = 5 := by sorry

end vector_magnitude_problem_l1822_182259


namespace percentage_equality_l1822_182228

theorem percentage_equality (x y : ℝ) (p : ℝ) (h1 : x / y = 4) (h2 : p / 100 * x = 20 / 100 * y) : p = 5 := by
  sorry

end percentage_equality_l1822_182228


namespace impossible_coin_probabilities_l1822_182222

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧ 
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end impossible_coin_probabilities_l1822_182222


namespace ken_kept_pencils_l1822_182207

def pencil_problem (initial_pencils : ℕ) (given_to_manny : ℕ) (extra_to_nilo : ℕ) : Prop :=
  let given_to_nilo : ℕ := given_to_manny + extra_to_nilo
  let total_given : ℕ := given_to_manny + given_to_nilo
  let kept : ℕ := initial_pencils - total_given
  kept = 20

theorem ken_kept_pencils :
  pencil_problem 50 10 10 :=
sorry

end ken_kept_pencils_l1822_182207


namespace two_isosceles_triangles_l1822_182211

/-- Represents a point in 2D space with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Checks if a triangle is isosceles based on its vertices -/
def isIsosceles (a b c : Point) : Bool :=
  let d1 := squaredDistance a b
  let d2 := squaredDistance b c
  let d3 := squaredDistance c a
  d1 = d2 || d2 = d3 || d3 = d1

/-- The four triangles given in the problem -/
def triangle1 : (Point × Point × Point) := ({x := 0, y := 0}, {x := 4, y := 0}, {x := 2, y := 3})
def triangle2 : (Point × Point × Point) := ({x := 1, y := 1}, {x := 1, y := 4}, {x := 4, y := 1})
def triangle3 : (Point × Point × Point) := ({x := 3, y := 0}, {x := 6, y := 0}, {x := 4, y := 3})
def triangle4 : (Point × Point × Point) := ({x := 5, y := 2}, {x := 8, y := 2}, {x := 7, y := 5})

theorem two_isosceles_triangles :
  let triangles := [triangle1, triangle2, triangle3, triangle4]
  (triangles.filter (fun t => isIsosceles t.1 t.2.1 t.2.2)).length = 2 := by
  sorry

end two_isosceles_triangles_l1822_182211


namespace work_project_solution_l1822_182251

/-- Represents a work project with a number of workers and days to complete. -/
structure WorkProject where
  workers : ℕ
  days : ℕ

/-- The condition when 2 workers are removed. -/
def condition1 (wp : WorkProject) : Prop :=
  (wp.workers - 2) * (wp.days + 4) = wp.workers * wp.days

/-- The condition when 3 workers are added. -/
def condition2 (wp : WorkProject) : Prop :=
  (wp.workers + 3) * (wp.days - 2) > wp.workers * wp.days

/-- The condition when 4 workers are added. -/
def condition3 (wp : WorkProject) : Prop :=
  (wp.workers + 4) * (wp.days - 3) > wp.workers * wp.days

/-- The main theorem stating the solution to the work project problem. -/
theorem work_project_solution :
  ∃ (wp : WorkProject),
    condition1 wp ∧
    condition2 wp ∧
    condition3 wp ∧
    wp.workers = 6 ∧
    wp.days = 8 := by
  sorry


end work_project_solution_l1822_182251


namespace expected_defective_60000_l1822_182250

/-- Represents a shipment of computer chips -/
structure Shipment where
  defective : ℕ
  total : ℕ

/-- Calculates the expected number of defective chips in a future shipment -/
def expectedDefective (shipments : List Shipment) (futureTotal : ℕ) : ℕ :=
  let totalDefective := shipments.map (·.defective) |>.sum
  let totalChips := shipments.map (·.total) |>.sum
  (totalDefective * futureTotal) / totalChips

/-- Theorem stating the expected number of defective chips in a shipment of 60,000 -/
theorem expected_defective_60000 (shipments : List Shipment) 
    (h1 : shipments = [
      ⟨2, 5000⟩, 
      ⟨4, 12000⟩, 
      ⟨2, 15000⟩, 
      ⟨4, 16000⟩
    ]) : 
    expectedDefective shipments 60000 = 15 := by
  sorry

end expected_defective_60000_l1822_182250


namespace least_three_digit_multiple_of_eight_l1822_182265

theorem least_three_digit_multiple_of_eight : ∀ n : ℕ, 
  n ≥ 100 ∧ n < 1000 ∧ n % 8 = 0 → n ≥ 104 := by
  sorry

end least_three_digit_multiple_of_eight_l1822_182265


namespace red_apples_sold_l1822_182206

theorem red_apples_sold (red green : ℕ) : 
  (red : ℚ) / green = 8 / 3 → 
  red + green = 44 → 
  red = 32 := by
sorry

end red_apples_sold_l1822_182206


namespace apple_distribution_l1822_182274

/-- Represents the number of apples Karen has at the end -/
def karens_final_apples (initial_apples : ℕ) : ℕ :=
  let after_first_transfer := initial_apples - 12
  (after_first_transfer - after_first_transfer / 2)

/-- Represents the number of apples Alphonso has at the end -/
def alphonsos_final_apples (initial_apples : ℕ) : ℕ :=
  let after_first_transfer := initial_apples + 12
  let karens_remaining := initial_apples - 12
  (after_first_transfer + karens_remaining / 2)

theorem apple_distribution (initial_apples : ℕ) 
  (h1 : initial_apples ≥ 12)
  (h2 : alphonsos_final_apples initial_apples = 4 * karens_final_apples initial_apples) :
  karens_final_apples initial_apples = 24 := by
  sorry

end apple_distribution_l1822_182274


namespace rachel_brownies_l1822_182224

theorem rachel_brownies (total : ℚ) : 
  (3 / 5 : ℚ) * total = 18 → total = 30 := by
  sorry

end rachel_brownies_l1822_182224


namespace spheres_in_cone_radius_l1822_182262

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Theorem stating the radius of spheres in a cone under specific conditions -/
theorem spheres_in_cone_radius (c : Cone) (s1 s2 : Sphere) : 
  c.baseRadius = 6 ∧ 
  c.height = 15 ∧ 
  s1.radius = s2.radius ∧
  -- The spheres are tangent to each other, the side, and the base of the cone
  -- (This condition is implicitly assumed in the statement)
  True →
  s1.radius = 12 * Real.sqrt 29 / 29 :=
sorry

end spheres_in_cone_radius_l1822_182262


namespace blood_expiration_theorem_l1822_182216

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Calculates the expiration date and time for a blood donation -/
def calculateExpirationDateTime (donationDateTime : DateTime) : DateTime :=
  sorry

/-- The number of seconds in a day -/
def secondsPerDay : ℕ := 86400

/-- The expiration time in seconds for a unit of blood -/
def bloodExpirationSeconds : ℕ := Nat.factorial 9

/-- Theorem stating that a blood donation made at 8 AM on January 15th 
    will expire on January 19th at 4:48 AM -/
theorem blood_expiration_theorem 
  (donationDateTime : DateTime)
  (h1 : donationDateTime.year = 2023)
  (h2 : donationDateTime.month = 1)
  (h3 : donationDateTime.day = 15)
  (h4 : donationDateTime.hour = 8)
  (h5 : donationDateTime.minute = 0) :
  let expirationDateTime := calculateExpirationDateTime donationDateTime
  expirationDateTime.year = 2023 ∧
  expirationDateTime.month = 1 ∧
  expirationDateTime.day = 19 ∧
  expirationDateTime.hour = 4 ∧
  expirationDateTime.minute = 48 :=
sorry

end blood_expiration_theorem_l1822_182216


namespace smallest_max_sum_l1822_182203

theorem smallest_max_sum (a b c d e f g : ℕ+) 
  (sum_eq : a + b + c + d + e + f + g = 2024) : 
  (∃ M : ℕ, 
    (M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g)))))) ∧ 
    (∀ M' : ℕ, 
      (M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g)))))) → 
      M ≤ M') ∧
    M = 338) := by
  sorry

end smallest_max_sum_l1822_182203


namespace back_wheel_perimeter_l1822_182239

/-- Given a front wheel with perimeter 30 that revolves 240 times, and a back wheel that
    revolves 360 times to cover the same distance, the perimeter of the back wheel is 20. -/
theorem back_wheel_perimeter (front_perimeter : ℝ) (front_revolutions : ℝ) 
  (back_revolutions : ℝ) (back_perimeter : ℝ) : 
  front_perimeter = 30 →
  front_revolutions = 240 →
  back_revolutions = 360 →
  front_perimeter * front_revolutions = back_perimeter * back_revolutions →
  back_perimeter = 20 := by
  sorry

end back_wheel_perimeter_l1822_182239


namespace quadratic_root_relation_l1822_182237

/-- Given two quadratic equations where the roots of one are three times the roots of the other, 
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -c ∧ s₁ * s₂ = a) ∧
               (3 * s₁ + 3 * s₂ = -a ∧ 9 * s₁ * s₂ = b)) →
  b / c = 27 := by
  sorry

end quadratic_root_relation_l1822_182237


namespace algebra_test_female_students_l1822_182270

theorem algebra_test_female_students 
  (total_average : ℝ)
  (male_count : ℕ)
  (male_average : ℝ)
  (female_average : ℝ)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 83)
  (h4 : female_average = 92) :
  ∃ (female_count : ℕ),
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 28 := by
  sorry

end algebra_test_female_students_l1822_182270


namespace theorem_1_theorem_2_theorem_3_l1822_182257

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 1

-- Theorem 1
theorem theorem_1 (a : ℝ) :
  f a 1 = 2 → a = 1 ∧ ∀ x, f 1 x ≥ -2 :=
sorry

-- Theorem 2
theorem theorem_2 (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 :=
sorry

-- Theorem 3
theorem theorem_3 (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ x, f a x ≤ f a y) → a ≤ -4 :=
sorry

end theorem_1_theorem_2_theorem_3_l1822_182257


namespace first_jump_exceeding_2000_l1822_182229

-- Define the jump sequence
def jump_sequence : ℕ → ℕ
  | 0 => 2  -- First jump (we use 0-based indexing here)
  | n + 1 => 2 * jump_sequence n + n

-- Define a function to check if a jump exceeds 2000 meters
def exceeds_2000 (n : ℕ) : Prop := jump_sequence n > 2000

-- Theorem statement
theorem first_jump_exceeding_2000 :
  (∀ m : ℕ, m < 14 → ¬(exceeds_2000 m)) ∧ exceeds_2000 14 := by sorry

end first_jump_exceeding_2000_l1822_182229


namespace equation_solution_l1822_182210

theorem equation_solution : ∃ x : ℚ, (5 + 3.5 * x = 2.1 * x - 25) ∧ (x = -150/7) := by
  sorry

end equation_solution_l1822_182210


namespace father_sees_boy_less_than_half_time_l1822_182240

/-- Represents a point on the perimeter of the square school -/
structure PerimeterPoint where
  side : Fin 4
  position : ℝ
  h_position : 0 ≤ position ∧ position ≤ 1

/-- Represents the movement of a person around the square school -/
structure Movement where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise
  start_point : PerimeterPoint

/-- Defines when two points are on the same side of the square -/
def on_same_side (p1 p2 : PerimeterPoint) : Prop :=
  p1.side = p2.side

/-- The boy's movement around the school -/
def boy_movement : Movement :=
  { speed := 10
  , direction := true  -- Always clockwise
  , start_point := { side := 0, position := 0, h_position := ⟨by norm_num, by norm_num⟩ } }

/-- The father's movement around the school -/
def father_movement : Movement :=
  { speed := 5
  , direction := true  -- Initial direction (can change)
  , start_point := { side := 0, position := 0, h_position := ⟨by norm_num, by norm_num⟩ } }

/-- Theorem stating that the father cannot see the boy for more than half the time -/
theorem father_sees_boy_less_than_half_time :
  ∀ (t : ℝ) (t_pos : 0 < t),
  ∃ (boy_pos father_pos : PerimeterPoint),
  (∀ τ : ℝ, 0 ≤ τ ∧ τ ≤ t →
    (on_same_side (boy_pos) (father_pos)) →
    (∃ (see_time : ℝ), see_time ≤ t / 2)) :=
sorry

end father_sees_boy_less_than_half_time_l1822_182240


namespace fraction_power_product_l1822_182295

theorem fraction_power_product : (1/3)^100 * 3^101 = 3 := by sorry

end fraction_power_product_l1822_182295


namespace gamma_delta_sum_l1822_182294

theorem gamma_delta_sum : 
  ∃ (γ δ : ℝ), ∀ x : ℝ, (x - γ) / (x + δ) = (x^2 - 90*x + 1980) / (x^2 + 60*x - 3240) → 
  γ + δ = 140 := by
  sorry

end gamma_delta_sum_l1822_182294


namespace unique_root_condition_l1822_182271

/-- The equation ln(x+a) - 4(x+a)^2 + a = 0 has a unique root at x = 3 if and only if a = (3 ln 2 + 1) / 2 -/
theorem unique_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.log (x + a) - 4 * (x + a)^2 + a = 0 ∧ x = 3) ↔ 
  a = (3 * Real.log 2 + 1) / 2 := by
sorry

end unique_root_condition_l1822_182271


namespace triangle_side_length_l1822_182200

/-- In a triangle XYZ, if ∠Z = 30°, ∠Y = 60°, and XZ = 12 units, then XY = 24 units. -/
theorem triangle_side_length (X Y Z : ℝ × ℝ) :
  let angle (A B C : ℝ × ℝ) : ℝ := sorry
  let distance (A B : ℝ × ℝ) : ℝ := sorry
  angle Z X Y = π / 6 →  -- 30°
  angle X Y Z = π / 3 →  -- 60°
  distance X Z = 12 →
  distance X Y = 24 :=
by sorry

end triangle_side_length_l1822_182200


namespace S_inter_T_eq_T_l1822_182298

/-- The set of odd integers -/
def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}

/-- The set of integers of the form 4n + 1 -/
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

/-- Theorem stating that the intersection of S and T is equal to T -/
theorem S_inter_T_eq_T : S ∩ T = T := by sorry

end S_inter_T_eq_T_l1822_182298


namespace equal_reading_time_l1822_182249

/-- The total number of pages in the novel -/
def total_pages : ℕ := 760

/-- Bob's reading speed in seconds per page -/
def bob_speed : ℕ := 45

/-- Chandra's reading speed in seconds per page -/
def chandra_speed : ℕ := 30

/-- The number of pages Chandra reads -/
def chandra_pages : ℕ := 456

/-- The number of pages Bob reads -/
def bob_pages : ℕ := total_pages - chandra_pages

theorem equal_reading_time : chandra_speed * chandra_pages = bob_speed * bob_pages := by
  sorry

end equal_reading_time_l1822_182249


namespace ice_cream_combinations_l1822_182276

theorem ice_cream_combinations (n : ℕ) (k : ℕ) : 
  n = 5 → k = 3 → Nat.choose (n + k - 1) (k - 1) = 21 := by
  sorry

end ice_cream_combinations_l1822_182276


namespace debate_team_group_size_l1822_182235

/-- The size of each group in a debate team -/
def group_size (num_boys num_girls num_groups : ℕ) : ℕ :=
  (num_boys + num_girls) / num_groups

/-- Theorem: The size of each group in the debate team is 7 -/
theorem debate_team_group_size :
  group_size 11 45 8 = 7 := by
  sorry

end debate_team_group_size_l1822_182235


namespace book_arrangement_count_l1822_182280

def num_books : ℕ := 7
def num_identical_books : ℕ := 3

theorem book_arrangement_count : 
  (num_books.factorial) / (num_identical_books.factorial) = 840 := by
  sorry

end book_arrangement_count_l1822_182280


namespace career_preference_representation_l1822_182246

theorem career_preference_representation (total_students : ℕ) 
  (male_ratio female_ratio : ℕ) (male_preference female_preference : ℕ) : 
  total_students = 30 →
  male_ratio = 2 →
  female_ratio = 3 →
  male_preference = 2 →
  female_preference = 3 →
  (((male_preference + female_preference : ℝ) / total_students) * 360 : ℝ) = 60 := by
  sorry

end career_preference_representation_l1822_182246


namespace initially_tagged_fish_l1822_182273

-- Define the total number of fish in the pond
def total_fish : ℕ := 750

-- Define the number of fish in the second catch
def second_catch : ℕ := 50

-- Define the number of tagged fish in the second catch
def tagged_in_second_catch : ℕ := 2

-- Define the ratio of tagged fish in the second catch
def tagged_ratio : ℚ := tagged_in_second_catch / second_catch

-- Theorem: The number of fish initially caught and tagged is 30
theorem initially_tagged_fish : 
  ∃ (T : ℕ), T = 30 ∧ (T : ℚ) / total_fish = tagged_ratio :=
sorry

end initially_tagged_fish_l1822_182273


namespace stating_raptors_score_l1822_182263

/-- 
Represents the scores of three teams in a cricket match, 
where the total score is 48 and one team wins over another by 18 points.
-/
structure CricketScores where
  eagles : ℕ
  raptors : ℕ
  hawks : ℕ
  total_is_48 : eagles + raptors + hawks = 48
  eagles_margin : eagles = raptors + 18

/-- 
Theorem stating that the Raptors' score is (30 - hawks) / 2
given the conditions of the cricket match.
-/
theorem raptors_score (scores : CricketScores) : 
  scores.raptors = (30 - scores.hawks) / 2 := by
  sorry

#check raptors_score

end stating_raptors_score_l1822_182263


namespace velocity_equal_distance_time_l1822_182218

/-- For uniform motion, the velocity that makes the distance equal to time is 1. -/
theorem velocity_equal_distance_time (s t v : ℝ) (h : s = v * t) (h2 : s = t) : v = 1 := by
  sorry

end velocity_equal_distance_time_l1822_182218


namespace subtracted_amount_l1822_182284

theorem subtracted_amount (number : ℝ) (result : ℝ) (amount : ℝ) : 
  number = 85 → 
  result = 23 → 
  0.4 * number - amount = result →
  amount = 11 := by
  sorry

end subtracted_amount_l1822_182284


namespace liars_guessing_game_theorem_l1822_182277

/-- The liar's guessing game -/
structure LiarsGuessingGame where
  k : ℕ+  -- The number of consecutive answers where at least one must be truthful
  n : ℕ+  -- The maximum size of the final guessing set

/-- A winning strategy for player B -/
def has_winning_strategy (game : LiarsGuessingGame) : Prop :=
  ∀ N : ℕ+, ∃ (strategy : ℕ+ → Finset ℕ+), 
    (∀ x : ℕ+, x ≤ N → x ∈ strategy N) ∧
    (Finset.card (strategy N) ≤ game.n)

/-- Main theorem about the liar's guessing game -/
theorem liars_guessing_game_theorem (game : LiarsGuessingGame) :
  (game.n ≥ 2^(game.k : ℕ) → has_winning_strategy game) ∧
  (∃ k : ℕ+, ∃ n : ℕ+, n ≥ (1.99 : ℝ)^(k : ℕ) ∧ 
    ¬(has_winning_strategy ⟨k, n⟩)) := by
  sorry

end liars_guessing_game_theorem_l1822_182277


namespace complement_intersection_theorem_l1822_182226

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 2}
def B : Set Int := {-1, 0, 1}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0, 1} := by sorry

end complement_intersection_theorem_l1822_182226


namespace shaded_square_area_fraction_l1822_182236

/-- The fraction of a 6x6 grid's area occupied by a square with vertices at midpoints of grid lines along the diagonal -/
theorem shaded_square_area_fraction (grid_size : ℕ) (shaded_square_side : ℝ) : 
  grid_size = 6 → 
  shaded_square_side = 1 / Real.sqrt 2 →
  (shaded_square_side^2) / (grid_size^2 : ℝ) = 1 / 72 := by
  sorry

end shaded_square_area_fraction_l1822_182236


namespace hyperbola_foci_distance_l1822_182213

/-- For a hyperbola with equation x^2/45 - y^2/5 = 1, the distance between its foci is 10√2 -/
theorem hyperbola_foci_distance :
  ∀ x y : ℝ,
  (x^2 / 45) - (y^2 / 5) = 1 →
  ∃ f₁ f₂ : ℝ × ℝ,
  (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 200 :=
by sorry

end hyperbola_foci_distance_l1822_182213


namespace concert_audience_fraction_l1822_182256

theorem concert_audience_fraction (total_audience : ℕ) 
  (second_band_fraction : ℚ) (h1 : total_audience = 150) 
  (h2 : second_band_fraction = 2/3) : 
  1 - second_band_fraction = 1/3 := by
  sorry

end concert_audience_fraction_l1822_182256


namespace pairball_playtime_l1822_182219

/-- Given a game of pairball with the following conditions:
  * There are 12 children participating.
  * Only 2 children can play at a time.
  * The game runs continuously for 120 minutes.
  * Every child has an equal amount of playtime.
  Prove that each child plays for 20 minutes. -/
theorem pairball_playtime (num_children : ℕ) (players_per_game : ℕ) (total_time : ℕ) 
  (h1 : num_children = 12)
  (h2 : players_per_game = 2)
  (h3 : total_time = 120)
  : (total_time * players_per_game) / num_children = 20 := by
  sorry

end pairball_playtime_l1822_182219


namespace parabola_vertex_l1822_182248

/-- The vertex of the parabola y = -3x^2 + 6x + 4 is (1, 7) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * x^2 + 6 * x + 4 → (1, 7) = (x, y) ∧ ∀ (x' : ℝ), y ≥ -3 * x'^2 + 6 * x' + 4 := by
  sorry

end parabola_vertex_l1822_182248


namespace ecosystem_probability_l1822_182291

theorem ecosystem_probability : ∀ (n : ℕ) (p q r : ℚ),
  n = 7 →
  p = 1 / 5 →
  q = 1 / 10 →
  r = 17 / 20 →
  p + q + r = 1 →
  (Nat.choose n 4 : ℚ) * p^4 * r^3 = 34391 / 1000000 :=
by sorry

end ecosystem_probability_l1822_182291


namespace square_function_is_even_l1822_182247

/-- The function f(x) = x^2 is an even function for all real numbers x. -/
theorem square_function_is_even : ∀ x : ℝ, (fun x => x^2) (-x) = (fun x => x^2) x := by
  sorry

end square_function_is_even_l1822_182247


namespace cents_ratio_randi_to_peter_l1822_182238

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The total cents Ray has -/
def ray_total_cents : ℕ := 175

/-- The cents Ray gives to Peter -/
def cents_to_peter : ℕ := 30

/-- The number of extra nickels Randi has compared to Peter -/
def extra_nickels_randi : ℕ := 6

/-- Theorem stating the ratio of cents given to Randi vs Peter -/
theorem cents_ratio_randi_to_peter :
  let peter_nickels := cents_to_peter / nickel_value
  let randi_nickels := peter_nickels + extra_nickels_randi
  let cents_to_randi := randi_nickels * nickel_value
  (cents_to_randi : ℚ) / cents_to_peter = 2 := by sorry

end cents_ratio_randi_to_peter_l1822_182238


namespace marc_total_spent_l1822_182268

/-- The total amount Marc spent on his purchases -/
def total_spent (model_car_price model_car_quantity paint_price paint_quantity
                 paintbrush_price paintbrush_quantity : ℕ) : ℕ :=
  model_car_price * model_car_quantity +
  paint_price * paint_quantity +
  paintbrush_price * paintbrush_quantity

/-- Theorem stating that Marc spent $160 in total -/
theorem marc_total_spent :
  total_spent 20 5 10 5 2 5 = 160 := by
  sorry

end marc_total_spent_l1822_182268


namespace linear_equation_exponent_sum_l1822_182209

/-- If x^(a-1) - 3y^(b-2) = 7 is a linear equation in x and y, then a + b = 5 -/
theorem linear_equation_exponent_sum (a b : ℝ) : 
  (∀ x y : ℝ, ∃ m n c : ℝ, x^(a-1) - 3*y^(b-2) = m*x + n*y + c) → a + b = 5 := by
  sorry

end linear_equation_exponent_sum_l1822_182209


namespace parallelogram_sum_l1822_182279

/-- A parallelogram with sides measuring 6y-2, 4x+5, 12y-10, and 8x+1 has x + y = 7/3 -/
theorem parallelogram_sum (x y : ℚ) : 
  (6 * y - 2 : ℚ) = (12 * y - 10 : ℚ) →
  (4 * x + 5 : ℚ) = (8 * x + 1 : ℚ) →
  x + y = 7/3 := by sorry

end parallelogram_sum_l1822_182279


namespace inequality_equivalence_l1822_182272

theorem inequality_equivalence (x : ℝ) : (x + 2) * (x - 9) < 0 ↔ -2 < x ∧ x < 9 := by
  sorry

end inequality_equivalence_l1822_182272


namespace complex_root_modulus_one_iff_divisible_by_six_l1822_182264

theorem complex_root_modulus_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (n + 2) % 6 = 0 := by
  sorry

end complex_root_modulus_one_iff_divisible_by_six_l1822_182264


namespace red_peaches_count_l1822_182243

theorem red_peaches_count (total_baskets : ℕ) (green_per_basket : ℕ) (total_peaches : ℕ) :
  total_baskets = 11 →
  green_per_basket = 18 →
  total_peaches = 308 →
  ∃ red_per_basket : ℕ,
    red_per_basket * total_baskets + green_per_basket * total_baskets = total_peaches ∧
    red_per_basket = 10 :=
by sorry

end red_peaches_count_l1822_182243


namespace f_property_l1822_182275

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 7

/-- Theorem stating that if f(-7) = -17, then f(7) = 31 -/
theorem f_property (a b : ℝ) (h : f a b (-7) = -17) : f a b 7 = 31 := by
  sorry

end f_property_l1822_182275


namespace max_surface_area_l1822_182288

/-- A 3D structure made of unit cubes -/
structure CubeStructure where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculate the surface area of a CubeStructure -/
def surface_area (s : CubeStructure) : ℕ :=
  2 * (s.width * s.length + s.width * s.height + s.length * s.height)

/-- The specific cube structure from the problem -/
def problem_structure : CubeStructure :=
  { width := 2, length := 4, height := 2 }

theorem max_surface_area :
  surface_area problem_structure = 48 :=
sorry

end max_surface_area_l1822_182288


namespace luncheon_cost_theorem_l1822_182227

/-- Cost of a luncheon item -/
structure LuncheonItem where
  sandwich : ℚ
  coffee : ℚ
  pie : ℚ

/-- Calculate the total cost of a luncheon -/
def luncheonCost (item : LuncheonItem) (s c p : ℕ) : ℚ :=
  s * item.sandwich + c * item.coffee + p * item.pie

theorem luncheon_cost_theorem (item : LuncheonItem) : 
  luncheonCost item 2 5 1 = 3 ∧
  luncheonCost item 5 8 1 = 27/5 ∧
  luncheonCost item 3 4 1 = 18/5 →
  luncheonCost item 2 2 1 = 13/5 := by
sorry

#eval (13 : ℚ) / 5  -- Expected output: 2.6

end luncheon_cost_theorem_l1822_182227


namespace second_company_base_rate_l1822_182233

/-- Represents the base rate and per-minute charge of a telephone company --/
structure TelephoneCharge where
  baseRate : ℝ
  perMinuteRate : ℝ

/-- Calculates the total charge for a given number of minutes --/
def totalCharge (tc : TelephoneCharge) (minutes : ℝ) : ℝ :=
  tc.baseRate + tc.perMinuteRate * minutes

theorem second_company_base_rate :
  let unitedTelephone : TelephoneCharge := { baseRate := 9, perMinuteRate := 0.25 }
  let otherCompany : TelephoneCharge := { baseRate := x, perMinuteRate := 0.20 }
  totalCharge unitedTelephone 60 = totalCharge otherCompany 60 →
  x = 12 := by
  sorry

end second_company_base_rate_l1822_182233


namespace binomial_19_13_l1822_182296

theorem binomial_19_13 : Nat.choose 19 13 = 27132 := by
  -- Given conditions
  have h1 : Nat.choose 20 13 = 77520 := by sorry
  have h2 : Nat.choose 20 14 = 38760 := by sorry
  have h3 : Nat.choose 18 12 = 18564 := by sorry
  
  -- Proof
  sorry

end binomial_19_13_l1822_182296


namespace arithmetic_sequence_problem_l1822_182258

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 3 = 3 →
  a 6 = 24 →
  a 9 = 45 := by
sorry

end arithmetic_sequence_problem_l1822_182258


namespace roots_are_zero_neg_five_and_a_l1822_182260

variable (a : ℝ)

def roots : Set ℝ := {x : ℝ | x * (x + 5)^2 * (a - x) = 0}

theorem roots_are_zero_neg_five_and_a : roots a = {0, -5, a} := by
  sorry

end roots_are_zero_neg_five_and_a_l1822_182260


namespace gondor_monday_phones_l1822_182241

/-- Represents the earnings and repair information for Gondor --/
structure GondorEarnings where
  phone_repair_cost : ℕ
  laptop_repair_cost : ℕ
  tuesday_phones : ℕ
  wednesday_laptops : ℕ
  thursday_laptops : ℕ
  total_earnings : ℕ

/-- Calculates the number of phones repaired on Monday --/
def monday_phones (g : GondorEarnings) : ℕ :=
  (g.total_earnings - (g.phone_repair_cost * g.tuesday_phones + 
   g.laptop_repair_cost * (g.wednesday_laptops + g.thursday_laptops))) / g.phone_repair_cost

/-- Theorem stating that Gondor repaired 3 phones on Monday --/
theorem gondor_monday_phones (g : GondorEarnings) 
  (h1 : g.phone_repair_cost = 10)
  (h2 : g.laptop_repair_cost = 20)
  (h3 : g.tuesday_phones = 5)
  (h4 : g.wednesday_laptops = 2)
  (h5 : g.thursday_laptops = 4)
  (h6 : g.total_earnings = 200) :
  monday_phones g = 3 := by
  sorry

end gondor_monday_phones_l1822_182241


namespace systematic_sampling_theorem_l1822_182287

/-- Represents a systematic sampling of examination rooms -/
structure SystematicSampling where
  totalRooms : Nat
  sampleSize : Nat
  firstRoom : Nat
  interval : Nat

/-- Checks if a room number is part of the systematic sample -/
def isSelected (s : SystematicSampling) (room : Nat) : Prop :=
  ∃ k : Nat, room = s.firstRoom + k * s.interval ∧ room ≤ s.totalRooms

/-- The set of selected room numbers in a systematic sampling -/
def selectedRooms (s : SystematicSampling) : Set Nat :=
  {room | isSelected s room}

theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.totalRooms = 64)
  (h2 : s.sampleSize = 8)
  (h3 : s.firstRoom = 5)
  (h4 : isSelected s 21)
  (h5 : s.interval = s.totalRooms / s.sampleSize) :
  selectedRooms s = {5, 13, 21, 29, 37, 45, 53, 61} := by
  sorry


end systematic_sampling_theorem_l1822_182287


namespace special_circle_equation_l1822_182244

/-- A circle with center on y = x, passing through origin, and chord of length 2 on x-axis -/
structure SpecialCircle where
  center : ℝ × ℝ
  center_on_line : center.2 = center.1
  passes_origin : (center.1 ^ 2 + center.2 ^ 2) = 2 * center.1 ^ 2
  chord_length : ∃ x : ℝ, (x - center.1) ^ 2 + center.2 ^ 2 = 2 * center.1 ^ 2 ∧ 
                           ((x - 1) - center.1) ^ 2 + center.2 ^ 2 = 2 * center.1 ^ 2

theorem special_circle_equation (c : SpecialCircle) : 
  (∀ x y : ℝ, (x - 1) ^ 2 + (y - 1) ^ 2 = 2) ∨ 
  (∀ x y : ℝ, (x + 1) ^ 2 + (y + 1) ^ 2 = 2) := by
  sorry

end special_circle_equation_l1822_182244


namespace tangent_line_intersection_l1822_182214

/-- The function f(x) = x³ - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x² + a -/
def g (a x : ℝ) : ℝ := x^2 + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g(x) -/
def g' (x : ℝ) : ℝ := 2 * x

/-- The tangent line of f at x₁ -/
def tangent_f (x₁ x : ℝ) : ℝ := f' x₁ * (x - x₁) + f x₁

/-- The tangent line of g at x₂ -/
def tangent_g (a x₂ x : ℝ) : ℝ := g' x₂ * (x - x₂) + g a x₂

theorem tangent_line_intersection (a : ℝ) :
  (∃ x₁ x₂ : ℝ, ∀ x : ℝ, tangent_f x₁ x = tangent_g a x₂ x) →
  (x₁ = -1 → a = 3) ∧ (a ≥ -1) := by sorry

end tangent_line_intersection_l1822_182214
