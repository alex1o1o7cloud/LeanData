import Mathlib

namespace NUMINAMATH_CALUDE_harrys_morning_routine_time_l4025_402581

/-- Harry's morning routine time calculation -/
theorem harrys_morning_routine_time :
  let buying_time : ℕ := 15
  let eating_time : ℕ := 2 * buying_time
  let total_time : ℕ := buying_time + eating_time
  total_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_harrys_morning_routine_time_l4025_402581


namespace NUMINAMATH_CALUDE_square_of_cube_of_smallest_prime_l4025_402572

def smallest_prime : ℕ := 2

theorem square_of_cube_of_smallest_prime : 
  (smallest_prime ^ 3) ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_of_cube_of_smallest_prime_l4025_402572


namespace NUMINAMATH_CALUDE_chord_segment_lengths_l4025_402563

theorem chord_segment_lengths (r : ℝ) (chord_length : ℝ) :
  r = 7 ∧ chord_length = 12 →
  ∃ (ak kb : ℝ),
    ak = 7 - Real.sqrt 13 ∧
    kb = 7 + Real.sqrt 13 ∧
    ak + kb = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_chord_segment_lengths_l4025_402563


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l4025_402551

theorem max_value_cos_sin (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (M : Real), M = (3 : Real) / 2 ∧
  ∀ φ, 0 < φ ∧ φ < π →
    Real.cos (φ / 2) * (2 - Real.sin φ) ≤ M ∧
    ∃ ψ, 0 < ψ ∧ ψ < π ∧ Real.cos (ψ / 2) * (2 - Real.sin ψ) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l4025_402551


namespace NUMINAMATH_CALUDE_cucumber_weight_problem_l4025_402570

/-- Proves that the initial weight of cucumbers is 100 pounds given the conditions -/
theorem cucumber_weight_problem (initial_water_percent : Real) 
                                 (final_water_percent : Real)
                                 (final_weight : Real) :
  initial_water_percent = 0.99 →
  final_water_percent = 0.98 →
  final_weight = 50 →
  ∃ (initial_weight : Real),
    initial_weight = 100 ∧
    (1 - initial_water_percent) * initial_weight = (1 - final_water_percent) * final_weight :=
by
  sorry

#check cucumber_weight_problem

end NUMINAMATH_CALUDE_cucumber_weight_problem_l4025_402570


namespace NUMINAMATH_CALUDE_problem_solution_l4025_402526

theorem problem_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 945 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4025_402526


namespace NUMINAMATH_CALUDE_evaluate_expression_l4025_402562

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 5
  (2*x - a + 4) = (a + 14) := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4025_402562


namespace NUMINAMATH_CALUDE_triangle_formation_l4025_402566

/-- Triangle inequality check for three sides -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 7 12 17 ∧
  ¬ can_form_triangle 3 3 7 ∧
  ¬ can_form_triangle 4 5 9 ∧
  ¬ can_form_triangle 5 8 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l4025_402566


namespace NUMINAMATH_CALUDE_marks_jump_height_l4025_402585

theorem marks_jump_height :
  ∀ (mark_height lisa_height jacob_height james_height : ℝ),
    lisa_height = 2 * mark_height →
    jacob_height = 2 * lisa_height →
    james_height = 16 →
    james_height = 2/3 * jacob_height →
    mark_height = 6 := by
  sorry

end NUMINAMATH_CALUDE_marks_jump_height_l4025_402585


namespace NUMINAMATH_CALUDE_discriminant_nonnegativity_l4025_402584

theorem discriminant_nonnegativity (x : ℤ) :
  x^2 * (49 - 40 * x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_nonnegativity_l4025_402584


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_plus_sqrt_81_over_2_l4025_402591

theorem sqrt_of_sqrt_81_plus_sqrt_81_over_2 : 
  Real.sqrt ((Real.sqrt 81 + Real.sqrt 81) / 2) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_plus_sqrt_81_over_2_l4025_402591


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l4025_402546

-- Problem 1
theorem inequality_solution_1 : 
  {x : ℝ | -x^2 + 3*x + 10 < 0} = {x : ℝ | x > 5 ∨ x < -2} := by sorry

-- Problem 2
theorem inequality_solution_2 (a : ℝ) : 
  {x : ℝ | x^2 - 2*a*x + (a-1)*(a+1) ≤ 0} = {x : ℝ | a-1 ≤ x ∧ x ≤ a+1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l4025_402546


namespace NUMINAMATH_CALUDE_symmetric_circle_l4025_402537

/-- Given a point P(a, b) symmetric to line l with symmetric point P'(b + 1, a - 1),
    and a circle C with equation x^2 + y^2 - 6x - 2y = 0,
    prove that the equation of the circle C' symmetric to C with respect to line l
    is (x - 2)^2 + (y - 2)^2 = 10 -/
theorem symmetric_circle (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let P' : ℝ × ℝ := (b + 1, a - 1)
  let C (x y : ℝ) := x^2 + y^2 - 6*x - 2*y = 0
  let C' (x y : ℝ) := (x - 2)^2 + (y - 2)^2 = 10
  (∀ x y, C x y ↔ C' y x) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_l4025_402537


namespace NUMINAMATH_CALUDE_negative_square_cubed_l4025_402565

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l4025_402565


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l4025_402504

/-- In a right triangle, if the hypotenuse c is 2 more than one leg a,
    then the square of the other leg b is equal to 4a + 4 -/
theorem right_triangle_leg_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ b^2 = 4*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l4025_402504


namespace NUMINAMATH_CALUDE_equation_solution_l4025_402553

theorem equation_solution :
  ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 55 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4025_402553


namespace NUMINAMATH_CALUDE_differential_savings_proof_l4025_402501

def calculate_differential_savings (income : ℝ) (old_rate : ℝ) (new_rate : ℝ) : ℝ :=
  income * (old_rate - new_rate)

theorem differential_savings_proof (income : ℝ) (old_rate : ℝ) (new_rate : ℝ) 
  (h1 : income = 48000)
  (h2 : old_rate = 0.45)
  (h3 : new_rate = 0.30) :
  calculate_differential_savings income old_rate new_rate = 7200 := by
  sorry

end NUMINAMATH_CALUDE_differential_savings_proof_l4025_402501


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l4025_402535

theorem largest_divisor_of_expression : 
  ∃ (x : ℕ), x = 18 ∧ 
  (∀ (y : ℕ), x ∣ (7^y + 12*y - 1)) ∧
  (∀ (z : ℕ), z > x → ∃ (w : ℕ), ¬(z ∣ (7^w + 12*w - 1))) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l4025_402535


namespace NUMINAMATH_CALUDE_vector_magnitude_l4025_402571

/-- Given two vectors a and b in R², if (a - 2b) is perpendicular to a, then the magnitude of b is √5. -/
theorem vector_magnitude (a b : ℝ × ℝ) (h : a = (-1, 3) ∧ b.1 = 1) :
  (a - 2 • b) • a = 0 → ‖b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l4025_402571


namespace NUMINAMATH_CALUDE_square_perimeter_l4025_402523

theorem square_perimeter (s : ℝ) (h : s * s = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l4025_402523


namespace NUMINAMATH_CALUDE_upperclassmen_sport_players_l4025_402594

/-- Represents the number of students who play a sport in a college --/
structure SportPlayers where
  total : ℕ
  freshmen : ℕ
  upperclassmen : ℕ
  freshmenPercent : ℚ
  upperclassmenPercent : ℚ
  totalNonPlayersPercent : ℚ

/-- Theorem stating that given the conditions, 383 upperclassmen play a sport --/
theorem upperclassmen_sport_players (sp : SportPlayers)
  (h1 : sp.total = 800)
  (h2 : sp.freshmenPercent = 35 / 100)
  (h3 : sp.upperclassmenPercent = 75 / 100)
  (h4 : sp.totalNonPlayersPercent = 395 / 1000)
  : sp.upperclassmen = 383 := by
  sorry

#check upperclassmen_sport_players

end NUMINAMATH_CALUDE_upperclassmen_sport_players_l4025_402594


namespace NUMINAMATH_CALUDE_greta_is_oldest_l4025_402536

-- Define the set of people
inductive Person : Type
| Ada : Person
| Darwyn : Person
| Max : Person
| Greta : Person
| James : Person

-- Define the age relation
def younger_than (a b : Person) : Prop := sorry

-- Define the conditions
axiom ada_younger_than_darwyn : younger_than Person.Ada Person.Darwyn
axiom max_younger_than_greta : younger_than Person.Max Person.Greta
axiom james_older_than_darwyn : younger_than Person.Darwyn Person.James
axiom max_same_age_as_james : ∀ p, younger_than Person.Max p ↔ younger_than Person.James p

-- Define the oldest person property
def is_oldest (p : Person) : Prop :=
  ∀ q : Person, q ≠ p → younger_than q p

-- Theorem statement
theorem greta_is_oldest : is_oldest Person.Greta := by
  sorry

end NUMINAMATH_CALUDE_greta_is_oldest_l4025_402536


namespace NUMINAMATH_CALUDE_train_crossing_time_l4025_402597

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 600 → 
  train_speed_kmh = 144 → 
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) → 
  crossing_time = 15 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4025_402597


namespace NUMINAMATH_CALUDE_unique_prime_triple_l4025_402596

theorem unique_prime_triple (p : ℕ) : 
  Prime p ∧ Prime (2 * p + 1) ∧ Prime (4 * p + 1) ↔ p = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l4025_402596


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l4025_402518

/-- The y-intercept of the line 6x + 10y = 40 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) : 6 * x + 10 * y = 40 → x = 0 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l4025_402518


namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l4025_402560

theorem root_sum_absolute_value (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2022*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 104 := by
sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l4025_402560


namespace NUMINAMATH_CALUDE_teds_age_l4025_402534

theorem teds_age (ted sally : ℝ) 
  (h1 : ted = 3 * sally - 20) 
  (h2 : ted + sally = 78) : 
  ted = 53.5 := by
sorry

end NUMINAMATH_CALUDE_teds_age_l4025_402534


namespace NUMINAMATH_CALUDE_dodecahedron_triangles_l4025_402564

/-- Represents a dodecahedron -/
structure Dodecahedron where
  num_faces : ℕ
  faces_are_pentagonal : num_faces = 12
  vertices_per_face : ℕ
  vertices_shared_by_three_faces : vertices_per_face = 3

/-- Calculates the number of vertices in a dodecahedron -/
def num_vertices (d : Dodecahedron) : ℕ := 20

/-- Calculates the number of triangles that can be formed using the vertices of a dodecahedron -/
def num_triangles (d : Dodecahedron) : ℕ := (num_vertices d).choose 3

/-- Theorem: The number of triangles that can be formed using the vertices of a dodecahedron is 1140 -/
theorem dodecahedron_triangles (d : Dodecahedron) : num_triangles d = 1140 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_triangles_l4025_402564


namespace NUMINAMATH_CALUDE_complex_fraction_magnitude_l4025_402556

theorem complex_fraction_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 1)
  (hw : Complex.abs w = 3)
  (hzw : Complex.abs (z + w) = 2) :
  Complex.abs (1 / z + 1 / w) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_magnitude_l4025_402556


namespace NUMINAMATH_CALUDE_aunt_may_milk_sales_l4025_402574

/-- Represents the milk production and sales for Aunt May's farm --/
structure MilkProduction where
  morning : ℕ  -- Morning milk production in gallons
  evening : ℕ  -- Evening milk production in gallons
  leftover : ℕ  -- Leftover milk from yesterday in gallons
  remaining : ℕ  -- Remaining milk after selling in gallons

/-- Calculates the amount of milk sold to the ice cream factory --/
def milk_sold (p : MilkProduction) : ℕ :=
  p.morning + p.evening + p.leftover - p.remaining

/-- Theorem stating the amount of milk sold to the ice cream factory --/
theorem aunt_may_milk_sales (p : MilkProduction)
  (h_morning : p.morning = 365)
  (h_evening : p.evening = 380)
  (h_leftover : p.leftover = 15)
  (h_remaining : p.remaining = 148) :
  milk_sold p = 612 := by
  sorry

#eval milk_sold { morning := 365, evening := 380, leftover := 15, remaining := 148 }

end NUMINAMATH_CALUDE_aunt_may_milk_sales_l4025_402574


namespace NUMINAMATH_CALUDE_shopping_cost_after_discount_l4025_402579

/-- Calculate the total cost after discount for a shopping trip --/
theorem shopping_cost_after_discount :
  let tshirt_cost : ℕ := 20
  let pants_cost : ℕ := 80
  let shoes_cost : ℕ := 150
  let discount_rate : ℚ := 1 / 10
  let tshirt_quantity : ℕ := 4
  let pants_quantity : ℕ := 3
  let shoes_quantity : ℕ := 2
  let total_cost_before_discount : ℕ := 
    tshirt_cost * tshirt_quantity + 
    pants_cost * pants_quantity + 
    shoes_cost * shoes_quantity
  let discount_amount : ℚ := discount_rate * total_cost_before_discount
  let total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount
  total_cost_after_discount = 558 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_after_discount_l4025_402579


namespace NUMINAMATH_CALUDE_cubic_diophantine_equation_solution_l4025_402561

theorem cubic_diophantine_equation_solution :
  ∀ x y : ℕ+, x^3 - y^3 = x * y + 61 → (x = 6 ∧ y = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_diophantine_equation_solution_l4025_402561


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4025_402525

/-- Given a hyperbola C and a parabola, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5 →  -- Focal distance
  (∃ k : ℝ, ∀ x : ℝ, (1/16 * x^2 + 1 - k*x = 0 → 
    (k = b/a ∨ k = -b/a))) →  -- Parabola tangent to asymptotes
  (∀ x y : ℝ, x^2 / 4 - y^2 = 1) :=  -- Conclusion: Specific hyperbola equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4025_402525


namespace NUMINAMATH_CALUDE_zero_points_count_l4025_402529

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem zero_points_count 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f π) 
  (h_shifted : ∀ x, f (x - π) = f (x + π) ∧ f (x - π) = f x) : 
  ∃ (S : Finset ℝ), S.card = 7 ∧ (∀ x ∈ S, f x = 0 ∧ x ∈ Set.Icc 0 8) ∧
    (∀ x ∈ Set.Icc 0 8, f x = 0 → x ∈ S) :=
sorry

end NUMINAMATH_CALUDE_zero_points_count_l4025_402529


namespace NUMINAMATH_CALUDE_not_equivalent_squared_and_equal_l4025_402568

variable {X : Type*}
variable (x : X)
variable (A B : X → ℝ)

theorem not_equivalent_squared_and_equal :
  ¬(∀ x, A x ^ 2 = B x ^ 2 ↔ A x = B x) :=
sorry

end NUMINAMATH_CALUDE_not_equivalent_squared_and_equal_l4025_402568


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l4025_402507

-- Define the hyperbola equation
def hyperbola_equation (x y α : ℝ) : Prop :=
  x^2 * Real.sin α + y^2 * Real.cos α = 1

-- Define the property of hyperbola with foci on y-axis
def foci_on_y_axis (α : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola_equation x y α ∧ Real.cos α > 0 ∧ Real.sin α < 0

-- Theorem statement
theorem angle_in_fourth_quadrant (α : ℝ) (h : foci_on_y_axis α) :
  α > -π/2 ∧ α < 0 :=
sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l4025_402507


namespace NUMINAMATH_CALUDE_polygon_sides_l4025_402543

theorem polygon_sides (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l4025_402543


namespace NUMINAMATH_CALUDE_average_cost_before_gratuity_l4025_402587

/-- Proves that for a group of 7 people with a total bill of $840 including 20% gratuity,
    the average cost per person before gratuity is $100. -/
theorem average_cost_before_gratuity 
  (num_people : ℕ) 
  (total_bill : ℝ) 
  (gratuity_rate : ℝ) :
  num_people = 7 →
  total_bill = 840 →
  gratuity_rate = 0.20 →
  (total_bill / (1 + gratuity_rate)) / num_people = 100 := by
sorry

end NUMINAMATH_CALUDE_average_cost_before_gratuity_l4025_402587


namespace NUMINAMATH_CALUDE_most_likely_outcome_l4025_402544

def n : ℕ := 5

def p_boy : ℚ := 1/2
def p_girl : ℚ := 1/2

def prob_all_same_gender : ℚ := p_boy^n + p_girl^n

def prob_three_two : ℚ := (Nat.choose n 3) * (p_boy^3 * p_girl^2 + p_boy^2 * p_girl^3)

theorem most_likely_outcome :
  prob_three_two > prob_all_same_gender ∧
  prob_three_two = 5/16 :=
sorry

end NUMINAMATH_CALUDE_most_likely_outcome_l4025_402544


namespace NUMINAMATH_CALUDE_peter_glasses_purchase_l4025_402598

/-- Represents the purchase of glasses by Peter --/
def glassesPurchase (smallCost largeCost initialMoney smallCount change : ℕ) : Prop :=
  ∃ (largeCount : ℕ),
    smallCost * smallCount + largeCost * largeCount = initialMoney - change

theorem peter_glasses_purchase :
  glassesPurchase 3 5 50 8 1 →
  ∃ (largeCount : ℕ), largeCount = 5 ∧ glassesPurchase 3 5 50 8 1 := by
  sorry

end NUMINAMATH_CALUDE_peter_glasses_purchase_l4025_402598


namespace NUMINAMATH_CALUDE_midpoint_distance_after_move_l4025_402528

/-- Given two points P(p,q) and Q(r,s) on a Cartesian plane with midpoint N(x,y),
    prove that after moving P 3 units right and 5 units up, and Q 5 units left and 3 units down,
    the distance between N and the new midpoint N' is √2. -/
theorem midpoint_distance_after_move (p q r s x y : ℝ) :
  x = (p + r) / 2 →
  y = (q + s) / 2 →
  let x' := (p + 3 + r - 5) / 2
  let y' := (q + 5 + s - 3) / 2
  Real.sqrt ((x - x')^2 + (y - y')^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_after_move_l4025_402528


namespace NUMINAMATH_CALUDE_melissa_games_played_l4025_402545

def points_per_game : ℕ := 12
def total_points : ℕ := 36

theorem melissa_games_played : 
  total_points / points_per_game = 3 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_played_l4025_402545


namespace NUMINAMATH_CALUDE_divisor_totient_sum_theorem_l4025_402592

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_totient_sum_theorem (n : ℕ) (c : ℕ) :
  (n > 0) →
  (divisor_count n + Nat.totient n = n + c) ↔
  ((c = 1 ∧ (n = 1 ∨ Nat.Prime n ∨ n = 4)) ∨
   (c = 0 ∧ (n = 6 ∨ n = 8 ∨ n = 9))) :=
by sorry

end NUMINAMATH_CALUDE_divisor_totient_sum_theorem_l4025_402592


namespace NUMINAMATH_CALUDE_coal_extraction_theorem_l4025_402578

/-- Represents the working time ratio and coal extraction for a year -/
structure YearData where
  ratio : Fin 4 → ℚ
  coal_extracted : ℚ

/-- Given the data for three years, calculate the total coal extraction for 4 months -/
def total_coal_extraction (year1 year2 year3 : YearData) : ℚ :=
  4 * (year1.coal_extracted * (year1.ratio 0 + year1.ratio 1 + year1.ratio 2 + year1.ratio 3) / 
      (year1.ratio 0 + year1.ratio 1 + year1.ratio 2 + year1.ratio 3) +
      year2.coal_extracted * (year2.ratio 0 + year2.ratio 1 + year2.ratio 2 + year2.ratio 3) / 
      (year2.ratio 0 + year2.ratio 1 + year2.ratio 2 + year2.ratio 3) +
      year3.coal_extracted * (year3.ratio 0 + year3.ratio 1 + year3.ratio 2 + year3.ratio 3) / 
      (year3.ratio 0 + year3.ratio 1 + year3.ratio 2 + year3.ratio 3)) / 3

theorem coal_extraction_theorem (year1 year2 year3 : YearData) 
  (h1 : year1.ratio 0 = 4 ∧ year1.ratio 1 = 1 ∧ year1.ratio 2 = 2 ∧ year1.ratio 3 = 5 ∧ year1.coal_extracted = 10)
  (h2 : year2.ratio 0 = 2 ∧ year2.ratio 1 = 3 ∧ year2.ratio 2 = 2 ∧ year2.ratio 3 = 1 ∧ year2.coal_extracted = 7)
  (h3 : year3.ratio 0 = 5 ∧ year3.ratio 1 = 2 ∧ year3.ratio 2 = 1 ∧ year3.ratio 3 = 4 ∧ year3.coal_extracted = 14) :
  total_coal_extraction year1 year2 year3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_coal_extraction_theorem_l4025_402578


namespace NUMINAMATH_CALUDE_point_on_x_axis_with_distance_l4025_402555

/-- A point P on the x-axis that is √30 distance from P₁(4,1,2) has x-coordinate 9 or -1 -/
theorem point_on_x_axis_with_distance (x : ℝ) :
  (x - 4)^2 + 1^2 + 2^2 = 30 → x = 9 ∨ x = -1 := by
  sorry

#check point_on_x_axis_with_distance

end NUMINAMATH_CALUDE_point_on_x_axis_with_distance_l4025_402555


namespace NUMINAMATH_CALUDE_recurring_decimal_ratio_l4025_402586

-- Define the recurring decimals
def recurring_81 : ℚ := 81 / 99
def recurring_54 : ℚ := 54 / 99

-- State the theorem
theorem recurring_decimal_ratio :
  recurring_81 / recurring_54 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_ratio_l4025_402586


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l4025_402527

/-- Given a cylinder with volume 15 cubic feet, prove that tripling its radius
    and halving its height results in a new volume of 67.5 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 15 → π * (3*r)^2 * (h/2) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l4025_402527


namespace NUMINAMATH_CALUDE_correct_score_is_even_l4025_402509

/-- Represents the scoring system for a math competition -/
structure ScoringSystem where
  correct : Int
  unanswered : Int
  incorrect : Int

/-- Represents the results of a class in the math competition -/
structure CompetitionResult where
  total_questions : Nat
  scoring : ScoringSystem
  first_calculation : Int
  second_calculation : Int

/-- Theorem stating that the correct total score must be even -/
theorem correct_score_is_even (result : CompetitionResult) 
  (h1 : result.scoring.correct = 3)
  (h2 : result.scoring.unanswered = 1)
  (h3 : result.scoring.incorrect = -1)
  (h4 : result.total_questions = 50)
  (h5 : result.first_calculation = 5734)
  (h6 : result.second_calculation = 5735)
  (h7 : result.first_calculation = 5734 ∨ result.second_calculation = 5734) :
  ∃ (n : Int), 2 * n = 5734 ∧ (result.first_calculation = 5734 ∨ result.second_calculation = 5734) :=
sorry

end NUMINAMATH_CALUDE_correct_score_is_even_l4025_402509


namespace NUMINAMATH_CALUDE_pawpaw_count_l4025_402550

/-- Represents the contents of fruit baskets -/
structure FruitBaskets where
  total_fruits : Nat
  num_baskets : Nat
  mangoes : Nat
  pears : Nat
  lemons : Nat
  kiwi : Nat
  pawpaws : Nat

/-- Theorem stating the number of pawpaws in one basket -/
theorem pawpaw_count (fb : FruitBaskets) 
  (h1 : fb.total_fruits = 58)
  (h2 : fb.num_baskets = 5)
  (h3 : fb.mangoes = 18)
  (h4 : fb.pears = 10)
  (h5 : fb.lemons = 9)
  (h6 : fb.kiwi = fb.lemons)
  (h7 : fb.total_fruits = fb.mangoes + fb.pears + fb.lemons + fb.kiwi + fb.pawpaws) :
  fb.pawpaws = 12 := by
  sorry

end NUMINAMATH_CALUDE_pawpaw_count_l4025_402550


namespace NUMINAMATH_CALUDE_smallest_integer_cube_root_l4025_402557

theorem smallest_integer_cube_root (m n : ℕ) (r : ℝ) : 
  (0 < n) →
  (0 < r) →
  (r < 1/100) →
  (m = ((n : ℝ) + r)^3) →
  (∀ k < m, ¬∃ (s : ℝ), 0 < s ∧ s < 1/100 ∧ (k : ℝ)^(1/3) = (k : ℝ) + s) →
  (n = 6) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_cube_root_l4025_402557


namespace NUMINAMATH_CALUDE_weight_loss_days_l4025_402559

/-- Calculates the number of days required to lose a given amount of weight
    under specific calorie intake and expenditure conditions. -/
def days_to_lose_weight (pounds_to_lose : ℕ) (calories_per_pound : ℕ) 
    (calories_burned_per_day : ℕ) (calories_eaten_per_day : ℕ) : ℕ :=
  let total_calories_to_burn := pounds_to_lose * calories_per_pound
  let net_calories_burned_per_day := calories_burned_per_day - calories_eaten_per_day
  total_calories_to_burn / net_calories_burned_per_day

/-- Theorem stating that it takes 35 days to lose 5 pounds under the given conditions -/
theorem weight_loss_days : 
  days_to_lose_weight 5 3500 2500 2000 = 35 := by
  sorry

#eval days_to_lose_weight 5 3500 2500 2000

end NUMINAMATH_CALUDE_weight_loss_days_l4025_402559


namespace NUMINAMATH_CALUDE_range_of_f_l4025_402533

def f (x : Int) : Int := (x - 1)^2 + 1

def domain : Set Int := {-1, 0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l4025_402533


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l4025_402576

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) 
  (added_alcohol : ℝ) (added_water : ℝ) : 
  initial_volume = 40 →
  initial_percentage = 5 →
  added_alcohol = 4.5 →
  added_water = 5.5 →
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  let final_percentage := (final_alcohol / final_volume) * 100
  final_percentage = 13 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l4025_402576


namespace NUMINAMATH_CALUDE_vector_equation_solution_l4025_402595

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are not collinear -/
def NotCollinear (a b : V) : Prop := ∀ (k : ℝ), k • a ≠ b

theorem vector_equation_solution
  (a b : V) (x : ℝ)
  (h_not_collinear : NotCollinear a b)
  (h_c : ∃ c : V, c = x • a + b)
  (h_d : ∃ d : V, d = a + (2*x - 1) • b)
  (h_collinear : ∃ (k : ℝ) (c d : V), c = x • a + b ∧ d = a + (2*x - 1) • b ∧ d = k • c) :
  x = 1 ∨ x = -1/2 := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l4025_402595


namespace NUMINAMATH_CALUDE_neil_charge_theorem_l4025_402552

def trim_cost : ℕ → ℝ := λ n => 5 * n
def shape_cost : ℕ → ℝ := λ n => 15 * n

theorem neil_charge_theorem (num_trim : ℕ) (num_shape : ℕ) 
  (h1 : num_trim = 30) (h2 : num_shape = 4) : 
  trim_cost num_trim + shape_cost num_shape = 210 := by
  sorry

end NUMINAMATH_CALUDE_neil_charge_theorem_l4025_402552


namespace NUMINAMATH_CALUDE_min_value_theorem_l4025_402506

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 3 * a) (h3 : 3 * b^2 ≤ a * (a + c)) (h4 : a * (a + c) ≤ 5 * b^2) :
  -18/5 ≤ (b - 2*c) / a ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    a₀ ≤ b₀ + c₀ ∧ b₀ + c₀ ≤ 3 * a₀ ∧ 3 * b₀^2 ≤ a₀ * (a₀ + c₀) ∧ a₀ * (a₀ + c₀) ≤ 5 * b₀^2 ∧
    (b₀ - 2*c₀) / a₀ = -18/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4025_402506


namespace NUMINAMATH_CALUDE_not_all_cells_marked_l4025_402530

/-- Represents a cell in the grid --/
structure Cell :=
  (x : Nat) (y : Nat)

/-- The grid of cells --/
def Grid := List Cell

/-- Checks if two cells are neighbors --/
def isNeighbor (c1 c2 : Cell) : Bool :=
  (c1.x = c2.x ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨
  (c1.y = c2.y ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1))

/-- Counts the number of marked neighbors for a cell --/
def countMarkedNeighbors (cell : Cell) (markedCells : List Cell) : Nat :=
  (markedCells.filter (isNeighbor cell)).length

/-- Spreads the marking to cells with at least two marked neighbors --/
def spread (grid : Grid) (markedCells : List Cell) : List Cell :=
  markedCells ++ (grid.filter (fun c => countMarkedNeighbors c markedCells ≥ 2))

/-- Creates a 10x10 grid --/
def createGrid : Grid :=
  List.range 10 >>= fun x => List.range 10 >>= fun y => [Cell.mk x y]

/-- The main theorem --/
theorem not_all_cells_marked (initialMarked : List Cell) 
  (h : initialMarked.length = 9) : 
  ∃ (finalMarked : List Cell), finalMarked = spread (createGrid) initialMarked ∧ 
  finalMarked.length < 100 := by
  sorry

end NUMINAMATH_CALUDE_not_all_cells_marked_l4025_402530


namespace NUMINAMATH_CALUDE_arithmetic_grid_solution_l4025_402593

/-- Represents a 7x1 arithmetic sequence -/
def RowSequence := Fin 7 → ℤ

/-- Represents a 4x1 arithmetic sequence -/
def ColumnSequence := Fin 4 → ℤ

/-- The problem setup -/
structure ArithmeticGrid :=
  (row : RowSequence)
  (col1 : ColumnSequence)
  (col2 : ColumnSequence)
  (is_arithmetic_row : ∀ i j k : Fin 7, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    row j - row i = row k - row j)
  (is_arithmetic_col1 : ∀ i j k : Fin 4, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    col1 j - col1 i = col1 k - col1 j)
  (is_arithmetic_col2 : ∀ i j k : Fin 4, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    col2 j - col2 i = col2 k - col2 j)
  (distinct_sequences : 
    (∀ i j : Fin 7, i ≠ j → row i - row j ≠ 0) ∧ 
    (∀ i j : Fin 4, i ≠ j → col1 i - col1 j ≠ 0) ∧ 
    (∀ i j : Fin 4, i ≠ j → col2 i - col2 j ≠ 0))
  (top_left : row 0 = 25)
  (middle_column : col1 1 = 12 ∧ col1 2 = 16)
  (bottom_right : col2 3 = -13)

/-- The main theorem -/
theorem arithmetic_grid_solution (grid : ArithmeticGrid) : grid.col2 0 = -16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_grid_solution_l4025_402593


namespace NUMINAMATH_CALUDE_angle_between_vectors_l4025_402522

def a : ℝ × ℝ := (3, 0)
def b : ℝ × ℝ := (-5, 5)

theorem angle_between_vectors : 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  θ = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l4025_402522


namespace NUMINAMATH_CALUDE_regular_pentagons_are_similar_l4025_402588

/-- A regular pentagon is a polygon with 5 sides of equal length and 5 equal angles -/
structure RegularPentagon where
  side_length : ℝ
  angle_measure : ℝ
  side_length_pos : side_length > 0
  angle_measure_pos : angle_measure > 0
  angle_sum : angle_measure * 5 = 540

/-- Two shapes are similar if they have the same shape but not necessarily the same size -/
def AreSimilar (p1 p2 : RegularPentagon) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ p2.side_length = k * p1.side_length

/-- Theorem: Any two regular pentagons are similar -/
theorem regular_pentagons_are_similar (p1 p2 : RegularPentagon) : AreSimilar p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagons_are_similar_l4025_402588


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l4025_402500

/-- Represents a die in the cube --/
structure Die where
  sides : Fin 6 → ℕ
  sum_opposite : ∀ i : Fin 3, sides i + sides (i + 3) = 7

/-- Represents the 4x4x4 cube made of dice --/
def Cube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube --/
def visible_sum (c : Cube) : ℕ := sorry

/-- Theorem stating the smallest possible sum of visible faces --/
theorem smallest_visible_sum (c : Cube) : 
  visible_sum c ≥ 136 ∧ ∃ c', visible_sum c' = 136 := by sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l4025_402500


namespace NUMINAMATH_CALUDE_division_problem_l4025_402549

theorem division_problem (n : ℕ) : n % 21 = 1 ∧ n / 21 = 9 → n = 190 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4025_402549


namespace NUMINAMATH_CALUDE_limit_rational_function_l4025_402505

/-- The limit of (x^2 + 2x - 3) / (x^3 + 4x^2 + 3x) as x approaches -3 is -2/3 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 3| ∧ |x + 3| < δ → 
    |(x^2 + 2*x - 3) / (x^3 + 4*x^2 + 3*x) + 2/3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_rational_function_l4025_402505


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l4025_402583

theorem triangle_cosine_inequality (A B C : ℝ) (h_non_obtuse : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ A + B + C = π) :
  (1 - Real.cos (2 * A)) * (1 - Real.cos (2 * B)) / (1 - Real.cos (2 * C)) +
  (1 - Real.cos (2 * C)) * (1 - Real.cos (2 * A)) / (1 - Real.cos (2 * B)) +
  (1 - Real.cos (2 * B)) * (1 - Real.cos (2 * C)) / (1 - Real.cos (2 * A)) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l4025_402583


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l4025_402577

theorem arithmetic_expression_evaluation :
  (∀ x y z : ℤ, x + y = z → (x = 6 ∧ y = -13 ∧ z = -7) ∨ (x = -5 ∧ y = -3 ∧ z = -8)) ∧
  (6 + (-13) = -7) ∧
  (6 + (-13) ≠ 7) ∧
  (6 + (-13) ≠ -19) ∧
  (-5 + (-3) ≠ 8) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l4025_402577


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l4025_402511

def M : Set ℤ := {0, 1}
def N : Set ℤ := {x | x^2 + x = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l4025_402511


namespace NUMINAMATH_CALUDE_arithmetic_progression_middle_term_l4025_402513

/-- If 2, b, and 10 form an arithmetic progression, then b = 6 -/
theorem arithmetic_progression_middle_term : 
  ∀ b : ℝ, (2 - b = b - 10) → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_middle_term_l4025_402513


namespace NUMINAMATH_CALUDE_smallest_number_in_ratio_l4025_402573

theorem smallest_number_in_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a * 5 = b * 3 →
  a * 7 = c * 3 →
  c = 56 →
  c - a = 32 →
  a = 24 := by sorry

end NUMINAMATH_CALUDE_smallest_number_in_ratio_l4025_402573


namespace NUMINAMATH_CALUDE_nala_seashells_l4025_402548

/-- The number of seashells Nala found on the first day -/
def first_day : ℕ := 5

/-- The total number of seashells Nala has -/
def total : ℕ := 36

/-- The number of seashells Nala found on the second day -/
def second_day : ℕ := 7

theorem nala_seashells : 
  first_day + second_day + 2 * (first_day + second_day) = total := by
  sorry

#check nala_seashells

end NUMINAMATH_CALUDE_nala_seashells_l4025_402548


namespace NUMINAMATH_CALUDE_common_divisor_and_remainder_l4025_402580

theorem common_divisor_and_remainder (a b c d : ℕ) : 
  a = 2613 ∧ b = 2243 ∧ c = 1503 ∧ d = 985 →
  ∃ (k : ℕ), k > 0 ∧ 
    k ∣ (a - b) ∧ k ∣ (b - c) ∧ k ∣ (c - d) ∧
    ∀ m : ℕ, m > k → ¬(m ∣ (a - b) ∧ m ∣ (b - c) ∧ m ∣ (c - d)) ∧
    a % k = b % k ∧ b % k = c % k ∧ c % k = d % k ∧
    k = 74 ∧ a % k = 23 :=
by sorry

end NUMINAMATH_CALUDE_common_divisor_and_remainder_l4025_402580


namespace NUMINAMATH_CALUDE_river_flow_rate_l4025_402531

/-- Given a river with specified dimensions and flow rate, calculate its flow speed in km/h -/
theorem river_flow_rate (depth : ℝ) (width : ℝ) (flow_volume : ℝ) :
  depth = 2 →
  width = 45 →
  flow_volume = 9000 →
  (flow_volume / (depth * width) / 1000 * 60) = 6 := by
  sorry

end NUMINAMATH_CALUDE_river_flow_rate_l4025_402531


namespace NUMINAMATH_CALUDE_range_of_a_given_false_proposition_l4025_402539

theorem range_of_a_given_false_proposition : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 ≤ 0) → 
  (∀ a : ℝ, -1 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_false_proposition_l4025_402539


namespace NUMINAMATH_CALUDE_remaining_coins_value_is_1030_l4025_402569

-- Define the initial number of coins
def initial_quarters : ℕ := 33
def initial_nickels : ℕ := 87
def initial_dimes : ℕ := 52

-- Define the number of borrowed coins
def borrowed_quarters : ℕ := 15
def borrowed_nickels : ℕ := 75

-- Define the value of each coin type in cents
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5
def dime_value : ℕ := 10

-- Define the function to calculate the total value of remaining coins
def remaining_coins_value : ℕ :=
  (initial_quarters * quarter_value + 
   initial_nickels * nickel_value + 
   initial_dimes * dime_value) - 
  (borrowed_quarters * quarter_value + 
   borrowed_nickels * nickel_value)

-- Theorem statement
theorem remaining_coins_value_is_1030 : 
  remaining_coins_value = 1030 := by sorry

end NUMINAMATH_CALUDE_remaining_coins_value_is_1030_l4025_402569


namespace NUMINAMATH_CALUDE_store_profit_calculation_l4025_402538

/-- Represents the pricing strategy and profit calculation for a store selling turtleneck sweaters. -/
theorem store_profit_calculation (C : ℝ) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_discount := 0.07
  
  let SP1 := C * (1 + initial_markup)
  let SP2 := SP1 * (1 + new_year_markup)
  let SPF := SP2 * (1 - february_discount)
  let profit := SPF - C
  
  profit / C = 0.395 := by sorry

end NUMINAMATH_CALUDE_store_profit_calculation_l4025_402538


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l4025_402589

/-- Given a set of observations with known properties, calculate the incorrect value --/
theorem incorrect_observation_value
  (n : ℕ)  -- Total number of observations
  (original_mean : ℝ)  -- Original mean of observations
  (correct_value : ℝ)  -- The correct value of the misrecorded observation
  (new_mean : ℝ)  -- New mean after correction
  (hn : n = 40)  -- There are 40 observations
  (hom : original_mean = 36)  -- The original mean was 36
  (hcv : correct_value = 34)  -- The correct value of the misrecorded observation is 34
  (hnm : new_mean = 36.45)  -- The new mean after correction is 36.45
  : ∃ (incorrect_value : ℝ), incorrect_value = 52 := by
  sorry


end NUMINAMATH_CALUDE_incorrect_observation_value_l4025_402589


namespace NUMINAMATH_CALUDE_expression_evaluation_l4025_402540

theorem expression_evaluation : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4025_402540


namespace NUMINAMATH_CALUDE_area_DEF_eq_sum_of_parts_l4025_402554

/-- Represents a triangle with an area -/
structure Triangle :=
  (area : ℝ)

/-- Represents the main triangle DEF -/
def DEF : Triangle := sorry

/-- Represents the point Q inside triangle DEF -/
def Q : Point := sorry

/-- Represents the three smaller triangles created by lines through Q -/
def u₁ : Triangle := { area := 16 }
def u₂ : Triangle := { area := 25 }
def u₃ : Triangle := { area := 36 }

/-- The theorem stating that the area of DEF is the sum of areas of u₁, u₂, and u₃ -/
theorem area_DEF_eq_sum_of_parts : DEF.area = u₁.area + u₂.area + u₃.area := by
  sorry

#check area_DEF_eq_sum_of_parts

end NUMINAMATH_CALUDE_area_DEF_eq_sum_of_parts_l4025_402554


namespace NUMINAMATH_CALUDE_smallest_three_digit_odd_multiple_of_three_l4025_402567

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def first_digit (n : ℕ) : ℕ := n / 100

theorem smallest_three_digit_odd_multiple_of_three :
  ∃ (n : ℕ), is_three_digit n ∧ 
             Odd (first_digit n) ∧ 
             n % 3 = 0 ∧
             (∀ m : ℕ, is_three_digit m ∧ Odd (first_digit m) ∧ m % 3 = 0 → n ≤ m) ∧
             n = 102 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_odd_multiple_of_three_l4025_402567


namespace NUMINAMATH_CALUDE_power_division_twentythree_l4025_402516

theorem power_division_twentythree : (23 : ℕ)^11 / (23 : ℕ)^8 = 12167 := by sorry

end NUMINAMATH_CALUDE_power_division_twentythree_l4025_402516


namespace NUMINAMATH_CALUDE_plan_d_more_economical_l4025_402575

/-- The cost per gigabyte for Plan C in cents -/
def plan_c_cost_per_gb : ℚ := 15

/-- The initial fee for Plan D in cents -/
def plan_d_initial_fee : ℚ := 3000

/-- The cost per gigabyte for Plan D in cents -/
def plan_d_cost_per_gb : ℚ := 8

/-- The minimum number of gigabytes for Plan D to be more economical -/
def min_gb_for_plan_d : ℕ := 429

theorem plan_d_more_economical :
  (∀ n : ℕ, n ≥ min_gb_for_plan_d →
    plan_d_initial_fee + n * plan_d_cost_per_gb < n * plan_c_cost_per_gb) ∧
  (∀ n : ℕ, n < min_gb_for_plan_d →
    plan_d_initial_fee + n * plan_d_cost_per_gb ≥ n * plan_c_cost_per_gb) :=
by sorry

end NUMINAMATH_CALUDE_plan_d_more_economical_l4025_402575


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l4025_402514

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l4025_402514


namespace NUMINAMATH_CALUDE_expression_proof_l4025_402502

theorem expression_proof (x₁ x₂ : ℝ) (E : ℝ → ℝ) :
  (∀ x, (x + 3)^2 / (E x) = 2) →
  x₁ - x₂ = 14 →
  ∃ x, E x = (x + 3)^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_proof_l4025_402502


namespace NUMINAMATH_CALUDE_stream_speed_in_rowing_problem_l4025_402508

/-- Proves that the speed of the stream is 20 kmph given the conditions of the rowing problem. -/
theorem stream_speed_in_rowing_problem (boat_speed : ℝ) (stream_speed : ℝ) :
  boat_speed = 60 →
  (∀ d : ℝ, d > 0 → d / (boat_speed - stream_speed) = 2 * (d / (boat_speed + stream_speed))) →
  stream_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_in_rowing_problem_l4025_402508


namespace NUMINAMATH_CALUDE_intersection_parallel_perpendicular_l4025_402590

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + y - 5 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y = 0
def line_l (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Define point P as the intersection of line1 and line2
def point_p : ℝ × ℝ := (2, 1)

-- Define the parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 3 * x - y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x + 3 * y - 5 = 0

theorem intersection_parallel_perpendicular :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ (x, y) = point_p) →
  (parallel_line (point_p.1) (point_p.2)) ∧
  (∀ (x y : ℝ), parallel_line x y → (y - point_p.2) = 3 * (x - point_p.1)) ∧
  (perpendicular_line (point_p.1) (point_p.2)) ∧
  (∀ (x y : ℝ), perpendicular_line x y → (y - point_p.2) = -(1/3) * (x - point_p.1)) :=
by sorry


end NUMINAMATH_CALUDE_intersection_parallel_perpendicular_l4025_402590


namespace NUMINAMATH_CALUDE_fifteen_people_handshakes_l4025_402524

/-- The number of handshakes in a group where each person shakes hands once with every other person -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 15 people, where each person shakes hands exactly once with every other person, the total number of handshakes is 105 -/
theorem fifteen_people_handshakes :
  handshakes 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_people_handshakes_l4025_402524


namespace NUMINAMATH_CALUDE_total_suit_cost_l4025_402512

/-- The cost of a suit given the following conditions:
  1. A jacket costs as much as trousers and a vest.
  2. A jacket and two pairs of trousers cost 175 dollars.
  3. Trousers and two vests cost 100 dollars. -/
def suit_cost (jacket trousers vest : ℝ) : Prop :=
  jacket = trousers + vest ∧
  jacket + 2 * trousers = 175 ∧
  trousers + 2 * vest = 100

/-- Theorem stating that the total cost of the suit is 150 dollars. -/
theorem total_suit_cost :
  ∀ (jacket trousers vest : ℝ),
    suit_cost jacket trousers vest →
    jacket + trousers + vest = 150 :=
by
  sorry

#check total_suit_cost

end NUMINAMATH_CALUDE_total_suit_cost_l4025_402512


namespace NUMINAMATH_CALUDE_density_not_vector_l4025_402547

/-- A type representing physical quantities --/
inductive PhysicalQuantity
| Buoyancy
| WindSpeed
| Displacement
| Density

/-- Definition of a vector --/
def isVector (q : PhysicalQuantity) : Prop :=
  ∃ (magnitude : ℝ) (direction : ℝ × ℝ × ℝ), True

/-- Theorem stating that density is not a vector --/
theorem density_not_vector : ¬ isVector PhysicalQuantity.Density := by
  sorry

end NUMINAMATH_CALUDE_density_not_vector_l4025_402547


namespace NUMINAMATH_CALUDE_kennel_cats_dogs_difference_l4025_402510

theorem kennel_cats_dogs_difference (num_dogs : ℕ) (num_cats : ℕ) : 
  num_dogs = 32 →
  num_cats * 4 = num_dogs * 3 →
  num_dogs - num_cats = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_kennel_cats_dogs_difference_l4025_402510


namespace NUMINAMATH_CALUDE_earthquake_relief_team_selection_l4025_402521

/-- The number of ways to select a team of 5 doctors from 3 specialties -/
def select_team (orthopedic neurosurgeon internist : ℕ) : ℕ :=
  let total := orthopedic + neurosurgeon + internist
  let team_size := 5
  -- Add the number of ways for each valid combination
  (orthopedic.choose 3 * neurosurgeon.choose 1 * internist.choose 1) +
  (orthopedic.choose 1 * neurosurgeon.choose 3 * internist.choose 1) +
  (orthopedic.choose 1 * neurosurgeon.choose 1 * internist.choose 3) +
  (orthopedic.choose 2 * neurosurgeon.choose 2 * internist.choose 1) +
  (orthopedic.choose 1 * neurosurgeon.choose 2 * internist.choose 2) +
  (orthopedic.choose 2 * neurosurgeon.choose 1 * internist.choose 2)

/-- Theorem: The number of ways to select 5 people from 3 orthopedic doctors, 
    4 neurosurgeons, and 5 internists, with at least one from each specialty, is 590 -/
theorem earthquake_relief_team_selection : select_team 3 4 5 = 590 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_relief_team_selection_l4025_402521


namespace NUMINAMATH_CALUDE_lineup_count_l4025_402517

def team_size : ℕ := 18

def lineup_positions : List String := ["goalkeeper", "center-back", "center-back", "left-back", "right-back", "midfielder", "midfielder", "midfielder"]

def number_of_lineups : ℕ :=
  team_size *
  (team_size - 1) * (team_size - 2) *
  (team_size - 3) *
  (team_size - 4) *
  (team_size - 5) * (team_size - 6) * (team_size - 7)

theorem lineup_count :
  number_of_lineups = 95414400 :=
by sorry

end NUMINAMATH_CALUDE_lineup_count_l4025_402517


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base7_l4025_402558

/-- The number of digits of a natural number in base 7 -/
def numDigitsBase7 (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log 7 n + 1

/-- Conversion from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ :=
  sorry

theorem largest_three_digit_square_base7 :
  ∃ N : ℕ, N = 45 ∧
  (∀ m : ℕ, m > N → numDigitsBase7 (m^2) > 3) ∧
  numDigitsBase7 (N^2) = 3 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base7_l4025_402558


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4025_402515

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and asymptote y = 2x,
    its eccentricity e is either √5 or √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y = 2*x) →
  ∃ e : ℝ, (e = Real.sqrt 5 ∨ e = Real.sqrt 5 / 2) ∧
    ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →
      e = Real.sqrt ((a^2 + b^2) / a^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4025_402515


namespace NUMINAMATH_CALUDE_p_and_q_sufficient_not_necessary_for_not_p_false_l4025_402542

theorem p_and_q_sufficient_not_necessary_for_not_p_false (p q : Prop) :
  (∃ (p q : Prop), (p ∧ q → ¬¬p) ∧ ¬(¬¬p → p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_p_and_q_sufficient_not_necessary_for_not_p_false_l4025_402542


namespace NUMINAMATH_CALUDE_fuel_mixture_theorem_l4025_402532

/-- Represents the state of the fuel tank -/
structure TankState where
  z : Rat  -- Amount of brand Z gasoline
  y : Rat  -- Amount of brand Y gasoline

/-- Fills the tank with brand Z gasoline -/
def fill_z (s : TankState) : TankState :=
  { z := s.z + (1 - s.z - s.y), y := s.y }

/-- Fills the tank with brand Y gasoline -/
def fill_y (s : TankState) : TankState :=
  { z := s.z, y := s.y + (1 - s.z - s.y) }

/-- Removes half of the fuel from the tank -/
def half_empty (s : TankState) : TankState :=
  { z := s.z / 2, y := s.y / 2 }

theorem fuel_mixture_theorem : 
  let s0 : TankState := { z := 0, y := 0 }
  let s1 := fill_z s0
  let s2 := fill_y (TankState.mk (3/4) 0)
  let s3 := fill_z (half_empty s2)
  let s4 := fill_y (half_empty s3)
  s4.z = 7/16 := by sorry

end NUMINAMATH_CALUDE_fuel_mixture_theorem_l4025_402532


namespace NUMINAMATH_CALUDE_flashlight_distance_difference_l4025_402519

/-- The visibility distance of Veronica's flashlight in feet -/
def veronica_distance : ℕ := 1000

/-- The visibility distance of Freddie's flashlight in feet -/
def freddie_distance : ℕ := 3 * veronica_distance

/-- The visibility distance of Velma's flashlight in feet -/
def velma_distance : ℕ := 5 * freddie_distance - 2000

/-- The difference in visibility distance between Velma's and Veronica's flashlights -/
theorem flashlight_distance_difference : velma_distance - veronica_distance = 12000 := by
  sorry

end NUMINAMATH_CALUDE_flashlight_distance_difference_l4025_402519


namespace NUMINAMATH_CALUDE_juice_bottle_savings_l4025_402599

/-- Represents the volume and cost of a juice bottle -/
structure Bottle :=
  (volume : ℕ)
  (cost : ℕ)

/-- Calculates the savings when buying a big bottle instead of equivalent small bottles -/
def calculate_savings (big : Bottle) (small : Bottle) : ℕ :=
  let small_bottles_needed := big.volume / small.volume
  let small_bottles_cost := small_bottles_needed * small.cost
  small_bottles_cost - big.cost

/-- Theorem stating the savings when buying a big bottle instead of equivalent small bottles -/
theorem juice_bottle_savings :
  let big_bottle := Bottle.mk 30 2700
  let small_bottle := Bottle.mk 6 600
  calculate_savings big_bottle small_bottle = 300 := by
sorry

end NUMINAMATH_CALUDE_juice_bottle_savings_l4025_402599


namespace NUMINAMATH_CALUDE_range_of_g_on_large_interval_l4025_402520

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def range_of (f : ℝ → ℝ) (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc a b, f x = y}

theorem range_of_g_on_large_interval
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_periodic : is_periodic f 1)
  (h_g_def : ∀ x, g x = f x + 2 * x)
  (h_range_small : range_of g 1 2 = Set.Icc (-1) 5) :
  range_of g (-2020) 2020 = Set.Icc (-4043) 4041 := by
sorry

end NUMINAMATH_CALUDE_range_of_g_on_large_interval_l4025_402520


namespace NUMINAMATH_CALUDE_number_of_pupils_l4025_402582

theorem number_of_pupils (total : ℕ) (parents : ℕ) (teachers : ℕ) 
  (h1 : total = 1541)
  (h2 : parents = 73)
  (h3 : teachers = 744) :
  total - (parents + teachers) = 724 := by
sorry

end NUMINAMATH_CALUDE_number_of_pupils_l4025_402582


namespace NUMINAMATH_CALUDE_boat_length_in_steps_l4025_402503

/-- Represents the scenario of Josie jogging alongside a moving boat --/
structure JosieAndBoat where
  josie_speed : ℝ
  boat_speed : ℝ
  boat_length : ℝ
  step_length : ℝ
  steps_forward : ℕ
  steps_backward : ℕ

/-- The conditions of the problem --/
def problem_conditions (scenario : JosieAndBoat) : Prop :=
  scenario.josie_speed > scenario.boat_speed ∧
  scenario.steps_forward = 130 ∧
  scenario.steps_backward = 70 ∧
  scenario.boat_length = scenario.step_length * 91

/-- The theorem to be proved --/
theorem boat_length_in_steps (scenario : JosieAndBoat) 
  (h : problem_conditions scenario) : 
  scenario.boat_length = scenario.step_length * 91 :=
sorry

end NUMINAMATH_CALUDE_boat_length_in_steps_l4025_402503


namespace NUMINAMATH_CALUDE_toby_friends_count_l4025_402541

theorem toby_friends_count (total : ℕ) (boys girls : ℕ) : 
  (boys : ℚ) / total = 55 / 100 →
  (girls : ℚ) / total = 30 / 100 →
  boys = 33 →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_toby_friends_count_l4025_402541
