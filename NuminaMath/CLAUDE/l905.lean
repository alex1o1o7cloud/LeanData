import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l905_90530

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25 :=          -- The shorter leg is 25 units
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l905_90530


namespace NUMINAMATH_CALUDE_leftover_value_is_correct_l905_90552

/-- Represents the number of coins in a roll --/
structure RollSize :=
  (quarters : Nat)
  (dimes : Nat)

/-- Represents the number of coins in a jar --/
structure JarContents :=
  (quarters : Nat)
  (dimes : Nat)

/-- Calculates the total value of leftover coins in dollars --/
def leftoverValue (rollSize : RollSize) (alice : JarContents) (bob : JarContents) : Rat :=
  let totalQuarters := alice.quarters + bob.quarters
  let totalDimes := alice.dimes + bob.dimes
  let leftoverQuarters := totalQuarters % rollSize.quarters
  let leftoverDimes := totalDimes % rollSize.dimes
  (leftoverQuarters * 25 + leftoverDimes * 10) / 100

theorem leftover_value_is_correct (rollSize : RollSize) (alice : JarContents) (bob : JarContents) :
  rollSize.quarters = 50 →
  rollSize.dimes = 60 →
  alice.quarters = 95 →
  alice.dimes = 184 →
  bob.quarters = 145 →
  bob.dimes = 312 →
  leftoverValue rollSize alice bob = 116/10 := by
  sorry

#eval leftoverValue ⟨50, 60⟩ ⟨95, 184⟩ ⟨145, 312⟩

end NUMINAMATH_CALUDE_leftover_value_is_correct_l905_90552


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l905_90519

-- Define the sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {-2, -1, 0, 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {-2, -1} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l905_90519


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l905_90504

/-- 
A right circular cylinder is inscribed in a right circular cone.
The cylinder's diameter equals its height.
The cone has a diameter of 8 and an altitude of 10.
The axes of the cylinder and the cone coincide.
-/
theorem inscribed_cylinder_radius (r : ℝ) : r = 20 / 9 :=
  let cone_diameter := 8
  let cone_altitude := 10
  let cylinder_height := 2 * r
  -- The cylinder's diameter equals its height
  have h1 : cylinder_height = 2 * r := rfl
  -- The cone has a diameter of 8 and an altitude of 10
  have h2 : cone_diameter = 8 := rfl
  have h3 : cone_altitude = 10 := rfl
  -- The axes of the cylinder and the cone coincide (implicit in the problem setup)
  sorry


end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l905_90504


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l905_90516

-- Problem 1
theorem problem_one (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

-- Problem 2
theorem problem_two (a b c x y z : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^x = b^y) (h5 : b^y = c^z)
  (h6 : 1/x + 1/y + 1/z = 0) : a * b * c = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l905_90516


namespace NUMINAMATH_CALUDE_book_sale_profit_l905_90551

/-- Calculates the difference between total selling price (including tax) and total purchase price (after discount) for books --/
theorem book_sale_profit (num_books : ℕ) (original_price discount_rate desired_price tax_rate : ℚ) : 
  num_books = 15 → 
  original_price = 11 → 
  discount_rate = 1/5 → 
  desired_price = 25 → 
  tax_rate = 1/10 → 
  (num_books * (desired_price * (1 + tax_rate))) - (num_books * (original_price * (1 - discount_rate))) = 280.5 := by
sorry

end NUMINAMATH_CALUDE_book_sale_profit_l905_90551


namespace NUMINAMATH_CALUDE_polynomial_root_property_l905_90507

/-- A polynomial of degree 4 with real coefficients -/
def PolynomialDegree4 (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

/-- The derivative of a polynomial of degree 4 -/
def DerivativePolynomialDegree4 (a b c : ℝ) (x : ℝ) : ℝ := 4*x^3 + 3*a*x^2 + 2*b*x + c

theorem polynomial_root_property (a b c d : ℝ) :
  let f := PolynomialDegree4 a b c d
  let f' := DerivativePolynomialDegree4 a b c
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) →
  (∃ w x y z : ℝ, w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z ∧
    f w = 0 ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) ∨
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x = 0 ∧ f y = 0 ∧ f z = 0 ∧
    (f' x = 0 ∨ f' y = 0 ∨ f' z = 0)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_property_l905_90507


namespace NUMINAMATH_CALUDE_function_symmetry_l905_90558

/-- Given a function f(x) = ax^4 - bx^2 + c - 1 where a, b, and c are real numbers,
    if f(2) = -1, then f(-2) = -1 -/
theorem function_symmetry (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^4 - b * x^2 + c - 1
  f 2 = -1 → f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l905_90558


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l905_90579

/-- The profit percentage of a cricket bat sale -/
def profit_percentage (selling_price profit : ℚ) : ℚ :=
  (profit / (selling_price - profit)) * 100

/-- Theorem: The profit percentage is 36% when a cricket bat is sold for $850 with a profit of $225 -/
theorem cricket_bat_profit_percentage :
  profit_percentage 850 225 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l905_90579


namespace NUMINAMATH_CALUDE_william_road_time_l905_90531

def departure_time : Nat := 7 * 60  -- 7:00 AM in minutes
def arrival_time : Nat := 20 * 60  -- 8:00 PM in minutes
def stop_durations : List Nat := [25, 10, 25]

def total_journey_time : Nat := arrival_time - departure_time
def total_stop_time : Nat := stop_durations.sum

theorem william_road_time :
  (total_journey_time - total_stop_time) / 60 = 12 := by sorry

end NUMINAMATH_CALUDE_william_road_time_l905_90531


namespace NUMINAMATH_CALUDE_animal_ages_l905_90582

theorem animal_ages (x : ℝ) 
  (h1 : 7 * (x - 3) = 2.5 * x - 3) : x + 2.5 * x = 14 := by
  sorry

end NUMINAMATH_CALUDE_animal_ages_l905_90582


namespace NUMINAMATH_CALUDE_one_is_monomial_l905_90535

/-- A monomial is an algebraic expression with only one term. -/
def IsMonomial (expr : ℕ) : Prop :=
  expr = 1 ∨ ∃ (base : ℕ) (exponent : ℕ), expr = base ^ exponent

/-- Theorem stating that 1 is a monomial. -/
theorem one_is_monomial : IsMonomial 1 := by sorry

end NUMINAMATH_CALUDE_one_is_monomial_l905_90535


namespace NUMINAMATH_CALUDE_factorization_proof_l905_90508

theorem factorization_proof (x y : ℝ) : x^2 - 2*x^2*y + x*y^2 = x*(x - 2*x*y + y^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l905_90508


namespace NUMINAMATH_CALUDE_region_characterization_l905_90534

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem region_characterization (x y : ℝ) :
  f x + f y ≤ 0 ∧ f x - f y ≥ 0 →
  (x - 3)^2 + (y - 3)^2 ≤ 8 ∧ (x - y)*(x + y - 6) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_region_characterization_l905_90534


namespace NUMINAMATH_CALUDE_puppies_count_l905_90563

/-- Calculates the number of puppies given the total food needed, mom's food consumption, and puppies' food consumption. -/
def number_of_puppies (total_food : ℚ) (mom_meal : ℚ) (mom_meals_per_day : ℕ) (puppy_meal : ℚ) (puppy_meals_per_day : ℕ) (days : ℕ) : ℕ :=
  let mom_food := mom_meal * mom_meals_per_day * days
  let puppy_food := total_food - mom_food
  let puppy_food_per_puppy := puppy_meal * puppy_meals_per_day * days
  (puppy_food / puppy_food_per_puppy).num.toNat

/-- Theorem stating that the number of puppies is 5 given the specified conditions. -/
theorem puppies_count : number_of_puppies 57 (3/2) 3 (1/2) 2 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_count_l905_90563


namespace NUMINAMATH_CALUDE_west_side_denial_percentage_l905_90554

theorem west_side_denial_percentage :
  let total_kids := 260
  let riverside_kids := 120
  let west_side_kids := 90
  let mountaintop_kids := 50
  let riverside_denied_percentage := 20
  let mountaintop_denied_percentage := 50
  let kids_admitted := 148
  
  let riverside_denied := riverside_kids * riverside_denied_percentage / 100
  let mountaintop_denied := mountaintop_kids * mountaintop_denied_percentage / 100
  let total_denied := total_kids - kids_admitted
  let west_side_denied := total_denied - riverside_denied - mountaintop_denied
  let west_side_denied_percentage := west_side_denied / west_side_kids * 100

  west_side_denied_percentage = 70 := by sorry

end NUMINAMATH_CALUDE_west_side_denial_percentage_l905_90554


namespace NUMINAMATH_CALUDE_balance_after_transfer_l905_90548

def initial_balance : ℝ := 400
def transfer_amount : ℝ := 90
def service_charge_rate : ℝ := 0.02

def final_balance : ℝ := initial_balance - (transfer_amount * (1 + service_charge_rate))

theorem balance_after_transfer :
  final_balance = 308.2 := by sorry

end NUMINAMATH_CALUDE_balance_after_transfer_l905_90548


namespace NUMINAMATH_CALUDE_jake_not_dropping_coffee_l905_90505

theorem jake_not_dropping_coffee (trip_probability : ℝ) (drop_given_trip_probability : ℝ) :
  trip_probability = 0.4 →
  drop_given_trip_probability = 0.25 →
  1 - trip_probability * drop_given_trip_probability = 0.9 := by
sorry

end NUMINAMATH_CALUDE_jake_not_dropping_coffee_l905_90505


namespace NUMINAMATH_CALUDE_sirokas_guests_l905_90586

/-- The number of guests Mrs. Široká was expecting -/
def num_guests : ℕ := 11

/-- The number of sandwiches in the first scenario -/
def sandwiches1 : ℕ := 25

/-- The number of sandwiches in the second scenario -/
def sandwiches2 : ℕ := 35

/-- The number of sandwiches in the final scenario -/
def sandwiches3 : ℕ := 52

theorem sirokas_guests :
  (sandwiches1 < 2 * num_guests + 3) ∧
  (sandwiches1 ≥ 2 * num_guests) ∧
  (sandwiches2 < 3 * num_guests + 4) ∧
  (sandwiches2 ≥ 3 * num_guests) ∧
  (sandwiches3 ≥ 4 * num_guests) ∧
  (sandwiches3 < 5 * num_guests) :=
by sorry

end NUMINAMATH_CALUDE_sirokas_guests_l905_90586


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l905_90536

theorem inscribed_squares_ratio : 
  ∀ x y : ℝ,
  (x > 0) →
  (y > 0) →
  (x^2 + x * 5 = 5 * 12) →
  (8/5 * y^2 + y^2 + 3/5 * y^2 = 10 * 2) →
  x / y = 96 / 85 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l905_90536


namespace NUMINAMATH_CALUDE_stating_acquaintance_group_relation_l905_90532

/-- 
A group of people with specific acquaintance relationships.
-/
structure AcquaintanceGroup where
  n : ℕ  -- Total number of people
  k : ℕ  -- Number of acquaintances per person
  l : ℕ  -- Number of common acquaintances for acquainted pairs
  m : ℕ  -- Number of common acquaintances for non-acquainted pairs
  k_lt_n : k < n  -- Each person is acquainted with fewer than the total number of people

/-- 
Theorem stating the relationship between the parameters of an AcquaintanceGroup.
-/
theorem acquaintance_group_relation (g : AcquaintanceGroup) : 
  g.m * (g.n - g.k - 1) = g.k * (g.k - g.l - 1) := by
  sorry

end NUMINAMATH_CALUDE_stating_acquaintance_group_relation_l905_90532


namespace NUMINAMATH_CALUDE_gcd_90_250_l905_90503

theorem gcd_90_250 : Nat.gcd 90 250 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_250_l905_90503


namespace NUMINAMATH_CALUDE_ellipse_equal_angles_point_l905_90545

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines the equation of the ellipse -/
def onEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Defines a chord passing through a given point -/
def isChord (e : Ellipse) (a b f : Point) : Prop :=
  onEllipse e a ∧ onEllipse e b ∧ ∃ t : ℝ, f = Point.mk (t * a.x + (1 - t) * b.x) (t * a.y + (1 - t) * b.y)

/-- Defines the property of equal angles -/
def equalAngles (p f a b : Point) : Prop :=
  (a.y - p.y) * (b.x - p.x) = (b.y - p.y) * (a.x - p.x)

/-- Main theorem statement -/
theorem ellipse_equal_angles_point :
  ∀ (e : Ellipse),
    e.a = 2 ∧ e.b = 1 →
    ∀ (f : Point),
      f.x = Real.sqrt 3 ∧ f.y = 0 →
      ∃! (p : Point),
        p.x > 0 ∧ p.y = 0 ∧
        (∀ (a b : Point), isChord e a b f → equalAngles p f a b) ∧
        p.x = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equal_angles_point_l905_90545


namespace NUMINAMATH_CALUDE_not_all_structure_diagrams_are_tree_shaped_l905_90566

/-- Represents a structure diagram -/
structure StructureDiagram where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Property: Elements show conceptual subordination and logical sequence -/
def shows_conceptual_subordination (sd : StructureDiagram) : Prop :=
  sorry

/-- Property: Can reflect relationships and overall characteristics -/
def reflects_relationships (sd : StructureDiagram) : Prop :=
  sorry

/-- Property: Can reflect system details thoroughly -/
def reflects_details (sd : StructureDiagram) : Prop :=
  sorry

/-- Property: Is tree-shaped -/
def is_tree_shaped (sd : StructureDiagram) : Prop :=
  sorry

/-- Theorem: Not all structure diagrams are tree-shaped -/
theorem not_all_structure_diagrams_are_tree_shaped :
  ¬ (∀ sd : StructureDiagram, is_tree_shaped sd) :=
sorry

end NUMINAMATH_CALUDE_not_all_structure_diagrams_are_tree_shaped_l905_90566


namespace NUMINAMATH_CALUDE_distinct_primes_in_product_l905_90590

theorem distinct_primes_in_product : 
  let n := 12 * 13 * 14 * 15
  Finset.card (Nat.factors n).toFinset = 5 := by
sorry

end NUMINAMATH_CALUDE_distinct_primes_in_product_l905_90590


namespace NUMINAMATH_CALUDE_lawnmower_value_drop_l905_90565

/-- Proves the percentage drop in lawnmower value after 6 months -/
theorem lawnmower_value_drop (initial_value : ℝ) (final_value : ℝ) (yearly_drop_percent : ℝ) :
  initial_value = 100 →
  final_value = 60 →
  yearly_drop_percent = 20 →
  final_value = initial_value * (1 - yearly_drop_percent / 100) →
  (initial_value - (final_value / (1 - yearly_drop_percent / 100))) / initial_value * 100 = 25 := by
  sorry

#check lawnmower_value_drop

end NUMINAMATH_CALUDE_lawnmower_value_drop_l905_90565


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l905_90572

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites and their y-coordinates are equal -/
def symmetric_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_implies_values (m n : ℝ) :
  symmetric_y_axis (-m, 3) (-5, n) → m = -5 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l905_90572


namespace NUMINAMATH_CALUDE_certain_number_proof_l905_90513

theorem certain_number_proof (n : ℕ) : 
  (∃ k : ℕ, n = k * 1 + 10 ∧ 2037 = k * 1 + 7) → n = 2040 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l905_90513


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l905_90561

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x < 0)) ↔ (∃ x : ℝ, x^2 - x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l905_90561


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l905_90515

theorem triangle_angle_theorem (a : ℝ) (x : ℝ) :
  (5 < a) → (a < 35) →
  (2 * a + 20) + (3 * a - 15) + x = 180 →
  x = 175 - 5 * a ∧
  ∃ (ε : ℝ), ε > 0 ∧ 35 - ε > a ∧
  max (2 * a + 20) (max (3 * a - 15) (175 - 5 * a)) = 88 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l905_90515


namespace NUMINAMATH_CALUDE_geometric_relations_l905_90595

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

end NUMINAMATH_CALUDE_geometric_relations_l905_90595


namespace NUMINAMATH_CALUDE_rectangle_triangle_same_area_altitude_l905_90573

theorem rectangle_triangle_same_area_altitude (h : ℝ) (w : ℝ) : 
  h > 0 →  -- Altitude is positive
  w > 0 →  -- Width is positive
  12 * h = 12 * w →  -- Areas are equal (12h for triangle, 12w for rectangle)
  w = h :=  -- Width of rectangle equals shared altitude
by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_same_area_altitude_l905_90573


namespace NUMINAMATH_CALUDE_f_one_root_f_odd_when_c_zero_f_symmetric_f_more_than_two_roots_l905_90540

-- Define the function f
def f (x b c : ℝ) : ℝ := |x| * x + b * x + c

-- Statement 1
theorem f_one_root (c : ℝ) (h : c > 0) : 
  ∃! x, f x 0 c = 0 := by sorry

-- Statement 2
theorem f_odd_when_c_zero (b : ℝ) :
  ∀ x, f (-x) b 0 = -(f x b 0) := by sorry

-- Statement 3
theorem f_symmetric (b c : ℝ) :
  ∀ x, f x b c = f (-x) b c := by sorry

-- Statement 4
theorem f_more_than_two_roots :
  ∃ b c, ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0 := by sorry

end NUMINAMATH_CALUDE_f_one_root_f_odd_when_c_zero_f_symmetric_f_more_than_two_roots_l905_90540


namespace NUMINAMATH_CALUDE_evaluate_expression_l905_90591

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/2) (hy : y = 1/3) (hz : z = -3) :
  (2*x)^2 * (y^2)^3 * z^2 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l905_90591


namespace NUMINAMATH_CALUDE_sequence_periodicity_l905_90593

theorem sequence_periodicity 
  (a b : ℕ → ℤ) 
  (h : ∀ n ≥ 3, (a n - a (n-1)) * (a n - a (n-2)) + (b n - b (n-1)) * (b n - b (n-2)) = 0) :
  ∃ k : ℕ+, a k = a (k + 2008) :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l905_90593


namespace NUMINAMATH_CALUDE_grace_marks_calculation_l905_90555

theorem grace_marks_calculation (num_students : ℕ) (initial_avg : ℚ) (final_avg : ℚ) 
  (h1 : num_students = 35)
  (h2 : initial_avg = 37)
  (h3 : final_avg = 40) :
  (num_students * final_avg - num_students * initial_avg) / num_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_grace_marks_calculation_l905_90555


namespace NUMINAMATH_CALUDE_socks_thrown_away_l905_90538

theorem socks_thrown_away (initial_socks : ℕ) (new_socks : ℕ) (final_socks : ℕ) : 
  initial_socks = 33 → new_socks = 13 → final_socks = 27 → 
  initial_socks - (final_socks - new_socks) = 19 := by
sorry

end NUMINAMATH_CALUDE_socks_thrown_away_l905_90538


namespace NUMINAMATH_CALUDE_dolly_initial_tickets_l905_90594

/- Define the number of rides for each attraction -/
def ferris_wheel_rides : Nat := 2
def roller_coaster_rides : Nat := 3
def log_ride_rides : Nat := 7

/- Define the ticket cost for each attraction -/
def ferris_wheel_cost : Nat := 2
def roller_coaster_cost : Nat := 5
def log_ride_cost : Nat := 1

/- Define the additional tickets needed -/
def additional_tickets : Nat := 6

/- Theorem to prove -/
theorem dolly_initial_tickets : 
  (ferris_wheel_rides * ferris_wheel_cost + 
   roller_coaster_rides * roller_coaster_cost + 
   log_ride_rides * log_ride_cost) - 
  additional_tickets = 20 := by
  sorry

end NUMINAMATH_CALUDE_dolly_initial_tickets_l905_90594


namespace NUMINAMATH_CALUDE_ashis_babji_height_comparison_l905_90520

theorem ashis_babji_height_comparison (ashis_height babji_height : ℝ) 
  (h : babji_height = ashis_height * (1 - 0.2)) : 
  (ashis_height - babji_height) / babji_height = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ashis_babji_height_comparison_l905_90520


namespace NUMINAMATH_CALUDE_randys_trees_l905_90518

/-- Proves that Randy has 5 less coconut trees compared to half the number of mango trees -/
theorem randys_trees (mango_trees : ℕ) (total_trees : ℕ) (coconut_trees : ℕ) : 
  mango_trees = 60 →
  total_trees = 85 →
  coconut_trees = total_trees - mango_trees →
  coconut_trees < mango_trees / 2 →
  mango_trees / 2 - coconut_trees = 5 := by
sorry

end NUMINAMATH_CALUDE_randys_trees_l905_90518


namespace NUMINAMATH_CALUDE_pie_distribution_l905_90544

theorem pie_distribution (total_slices : ℕ) (carl_portion : ℚ) (nancy_slices : ℕ) :
  total_slices = 8 →
  carl_portion = 1 / 4 →
  nancy_slices = 2 →
  (total_slices - carl_portion * total_slices - nancy_slices : ℚ) / total_slices = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_pie_distribution_l905_90544


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l905_90587

/-- Proves that in a triangle where two angles sum to 7/6 of a right angle,
    and one of these angles is 36° larger than the other,
    the largest angle in the triangle is 75°. -/
theorem largest_angle_in_special_triangle : 
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  a + b = 105 →
  b = a + 36 →
  max a (max b c) = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l905_90587


namespace NUMINAMATH_CALUDE_ping_pong_paddles_sold_l905_90502

/-- Given the total sales and average price per pair of ping pong paddles,
    prove the number of pairs sold. -/
theorem ping_pong_paddles_sold
  (total_sales : ℝ)
  (avg_price : ℝ)
  (h1 : total_sales = 735)
  (h2 : avg_price = 9.8) :
  total_sales / avg_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_paddles_sold_l905_90502


namespace NUMINAMATH_CALUDE_min_points_tenth_game_l905_90543

def points_four_games : List ℕ := [18, 22, 15, 19]

def average_greater_than_19 (total_points : ℕ) : Prop :=
  (total_points : ℚ) / 10 > 19

theorem min_points_tenth_game 
  (h1 : (points_four_games.sum : ℚ) / 4 > (List.sum (List.take 6 points_four_games) : ℚ) / 6)
  (h2 : ∃ (p : ℕ), average_greater_than_19 (points_four_games.sum + List.sum (List.take 6 points_four_games) + p)) :
  ∃ (p : ℕ), p ≥ 9 ∧ average_greater_than_19 (points_four_games.sum + List.sum (List.take 6 points_four_games) + p) ∧
  ∀ (q : ℕ), q < 9 → ¬average_greater_than_19 (points_four_games.sum + List.sum (List.take 6 points_four_games) + q) :=
sorry

end NUMINAMATH_CALUDE_min_points_tenth_game_l905_90543


namespace NUMINAMATH_CALUDE_seokgi_paper_usage_l905_90550

theorem seokgi_paper_usage (total : ℕ) (used : ℕ) (remaining : ℕ) : 
  total = 82 ∧ 
  remaining = total - used ∧ 
  remaining = used - 6 → 
  used = 44 := by sorry

end NUMINAMATH_CALUDE_seokgi_paper_usage_l905_90550


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l905_90578

def total_students : ℕ := 250
def drama_club : ℕ := 80
def science_club : ℕ := 120
def either_or_both : ℕ := 180

theorem students_in_both_clubs :
  ∃ (both : ℕ), both = drama_club + science_club - either_or_both ∧ both = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l905_90578


namespace NUMINAMATH_CALUDE_rocketry_club_theorem_l905_90541

theorem rocketry_club_theorem (total_students : ℕ) 
  (nails_neq_bolts : ℕ) (screws_eq_nails : ℕ) :
  total_students = 40 →
  nails_neq_bolts = 15 →
  screws_eq_nails = 10 →
  ∃ (screws_neq_bolts : ℕ), screws_neq_bolts ≥ 15 ∧
    screws_neq_bolts ≤ total_students - screws_eq_nails :=
by sorry

end NUMINAMATH_CALUDE_rocketry_club_theorem_l905_90541


namespace NUMINAMATH_CALUDE_intersection_slope_inequality_l905_90524

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + Real.log x)

-- Define the derivative of f
def f' (x : ℝ) : ℝ := Real.log x + 2

-- Theorem statement
theorem intersection_slope_inequality (x₁ x₂ k : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) 
  (h3 : k = (f' x₂ - f' x₁) / (x₂ - x₁)) : 
  x₁ < 1 / k ∧ 1 / k < x₂ := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_slope_inequality_l905_90524


namespace NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l905_90584

/-- Given an n-th degree polynomial P(x) such that P(k) = 1 / C_n^k for k = 0, 1, 2, ..., n,
    prove that P(n+1) = 0 if n is odd and P(n+1) = 1 if n is even. -/
theorem polynomial_value_at_n_plus_one (n : ℕ) (P : ℝ → ℝ) :
  (∀ k : ℕ, k ≤ n → P k = 1 / (n.choose k)) →
  P (n + 1) = if n % 2 = 1 then 0 else 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l905_90584


namespace NUMINAMATH_CALUDE_negation_existence_lt_one_l905_90514

theorem negation_existence_lt_one :
  (¬ ∃ x : ℝ, x < 1) ↔ (∀ x : ℝ, x ≥ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_lt_one_l905_90514


namespace NUMINAMATH_CALUDE_circle_through_points_tangent_line_through_D_tangent_touches_circle_l905_90597

-- Define the points
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 4)
def D : ℝ × ℝ := (-1, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 5

-- Define the tangent line equation
def tangent_line_equation (x y : ℝ) : Prop :=
  2*x + y = 0

-- Theorem for the circle equation
theorem circle_through_points :
  circle_equation A.1 A.2 ∧
  circle_equation B.1 B.2 ∧
  circle_equation C.1 C.2 := by sorry

-- Theorem for the tangent line
theorem tangent_line_through_D :
  tangent_line_equation D.1 D.2 := by sorry

-- Theorem that the tangent line touches the circle at exactly one point
theorem tangent_touches_circle :
  ∃! (x y : ℝ), circle_equation x y ∧ tangent_line_equation x y := by sorry

end NUMINAMATH_CALUDE_circle_through_points_tangent_line_through_D_tangent_touches_circle_l905_90597


namespace NUMINAMATH_CALUDE_salary_increase_with_manager_l905_90570

/-- Proves that adding a manager's salary increases the average salary by 150 rupees. -/
theorem salary_increase_with_manager 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 15 → 
  avg_salary = 1800 → 
  manager_salary = 4200 → 
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 150 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_with_manager_l905_90570


namespace NUMINAMATH_CALUDE_unique_condition_result_l905_90574

theorem unique_condition_result : ∃ (a b c : ℕ),
  ({a, b, c} : Set ℕ) = {0, 1, 2} ∧
  (((a ≠ 2) ∧ (b ≠ 2) ∧ (c = 0)) ∨
   ((a ≠ 2) ∧ (b = 2) ∧ (c = 0)) ∨
   ((a = 2) ∧ (b ≠ 2) ∧ (c ≠ 0))) →
  100 * a + 10 * b + c = 201 :=
by sorry

end NUMINAMATH_CALUDE_unique_condition_result_l905_90574


namespace NUMINAMATH_CALUDE_max_projection_equals_face_area_l905_90596

/-- A tetrahedron with two adjacent isosceles right triangle faces -/
structure Tetrahedron where
  /-- Length of the hypotenuse of the isosceles right triangle faces -/
  hypotenuse_length : ℝ
  /-- Dihedral angle between the two adjacent isosceles right triangle faces -/
  dihedral_angle : ℝ
  /-- Assumption that the hypotenuse length is 2 -/
  hypotenuse_is_two : hypotenuse_length = 2
  /-- Assumption that the dihedral angle is 60 degrees (π/3 radians) -/
  angle_is_sixty_degrees : dihedral_angle = π / 3

/-- The area of one isosceles right triangle face of the tetrahedron -/
def face_area (t : Tetrahedron) : ℝ := 1

/-- The maximum area of the projection of the rotating tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ := 1

/-- Theorem stating that the maximum projection area equals the face area -/
theorem max_projection_equals_face_area (t : Tetrahedron) :
  max_projection_area t = face_area t := by sorry

end NUMINAMATH_CALUDE_max_projection_equals_face_area_l905_90596


namespace NUMINAMATH_CALUDE_cookies_given_to_cousin_l905_90559

theorem cookies_given_to_cousin (initial_boxes : ℕ) (brother_boxes : ℕ) (sister_boxes : ℕ) (self_boxes : ℕ) :
  initial_boxes = 45 →
  brother_boxes = 12 →
  sister_boxes = 9 →
  self_boxes = 17 →
  initial_boxes - brother_boxes - sister_boxes - self_boxes = 7 :=
by sorry

end NUMINAMATH_CALUDE_cookies_given_to_cousin_l905_90559


namespace NUMINAMATH_CALUDE_percentage_speaking_both_truth_and_lies_l905_90537

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

end NUMINAMATH_CALUDE_percentage_speaking_both_truth_and_lies_l905_90537


namespace NUMINAMATH_CALUDE_uncle_dave_ice_cream_l905_90542

/-- The number of ice cream sandwiches Uncle Dave bought -/
def total_ice_cream_sandwiches : ℕ := sorry

/-- The number of Uncle Dave's nieces -/
def number_of_nieces : ℕ := 11

/-- The number of ice cream sandwiches each niece would get -/
def sandwiches_per_niece : ℕ := 13

/-- Theorem stating that the total number of ice cream sandwiches is 143 -/
theorem uncle_dave_ice_cream : total_ice_cream_sandwiches = number_of_nieces * sandwiches_per_niece := by
  sorry

end NUMINAMATH_CALUDE_uncle_dave_ice_cream_l905_90542


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l905_90564

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l905_90564


namespace NUMINAMATH_CALUDE_min_value_of_complex_sum_l905_90509

theorem min_value_of_complex_sum (z : ℂ) (h : Complex.abs (z + Complex.I) + Complex.abs (z - Complex.I) = 2) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ w : ℂ, Complex.abs (w + Complex.I) + Complex.abs (w - Complex.I) = 2 →
    Complex.abs (z + Complex.I + 1) ≤ Complex.abs (w + Complex.I + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_complex_sum_l905_90509


namespace NUMINAMATH_CALUDE_uncertain_mushrooms_l905_90575

theorem uncertain_mushrooms (total : ℕ) (safe : ℕ) (poisonous : ℕ) (uncertain : ℕ) : 
  total = 32 → 
  safe = 9 → 
  poisonous = 2 * safe → 
  total = safe + poisonous + uncertain → 
  uncertain = 5 := by
sorry

end NUMINAMATH_CALUDE_uncertain_mushrooms_l905_90575


namespace NUMINAMATH_CALUDE_sequence_ratio_theorem_l905_90568

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = q * b n

/-- Theorem: Given conditions on arithmetic and geometric sequences, prove the ratio -/
theorem sequence_ratio_theorem (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  a 1 = b 1 ∧ a 3 = b 2 ∧ a 7 = b 3 →
  (b 3 + b 4) / (b 4 + b 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_theorem_l905_90568


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l905_90571

theorem arithmetic_simplification :
  -11 - (-8) + (-13) + 12 = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l905_90571


namespace NUMINAMATH_CALUDE_mass_of_man_is_80kg_l905_90560

/-- The mass of a man who causes a boat to sink by a certain depth -/
def mass_of_man (boat_length boat_breadth sinking_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sinking_depth * water_density

/-- Theorem stating that the mass of the man is 80 kg -/
theorem mass_of_man_is_80kg :
  mass_of_man 4 2 0.01 1000 = 80 := by sorry

end NUMINAMATH_CALUDE_mass_of_man_is_80kg_l905_90560


namespace NUMINAMATH_CALUDE_atom_particle_count_l905_90588

/-- Represents an atom with a given number of protons and mass number -/
structure Atom where
  protons : ℕ
  massNumber : ℕ

/-- Calculates the total number of fundamental particles in an atom -/
def totalParticles (a : Atom) : ℕ :=
  a.protons + (a.massNumber - a.protons) + a.protons

/-- Theorem: The total number of fundamental particles in an atom with 9 protons and mass number 19 is 28 -/
theorem atom_particle_count :
  let a : Atom := { protons := 9, massNumber := 19 }
  totalParticles a = 28 := by
  sorry


end NUMINAMATH_CALUDE_atom_particle_count_l905_90588


namespace NUMINAMATH_CALUDE_simplify_expression_l905_90576

theorem simplify_expression : 
  Real.sqrt 6 * 6^(1/2) + 18 / 3 * 4 - (2 + 2)^(5/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l905_90576


namespace NUMINAMATH_CALUDE_sum_of_abs_roots_l905_90562

theorem sum_of_abs_roots : ∃ (r₁ r₂ r₃ r₄ : ℝ),
  (∀ x : ℝ, x^4 - 4*x^3 - 4*x^2 + 16*x - 8 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄)) ∧
  |r₁| + |r₂| + |r₃| + |r₄| = 2 + 2 * Real.sqrt 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_roots_l905_90562


namespace NUMINAMATH_CALUDE_tape_area_calculation_l905_90581

theorem tape_area_calculation (width : ℝ) (length : ℝ) (num_pieces : ℕ) (overlap : ℝ) 
  (h_width : width = 9.4)
  (h_length : length = 3.7)
  (h_num_pieces : num_pieces = 15)
  (h_overlap : overlap = 0.6) :
  let single_area := width * length
  let total_area_no_overlap := num_pieces * single_area
  let overlap_area := overlap * length
  let total_overlap_area := (num_pieces - 1) * overlap_area
  let total_area := total_area_no_overlap - total_overlap_area
  total_area = 490.62 := by sorry

end NUMINAMATH_CALUDE_tape_area_calculation_l905_90581


namespace NUMINAMATH_CALUDE_complement_S_union_T_equals_interval_l905_90521

open Set Real

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem complement_S_union_T_equals_interval :
  (𝒰 \ S) ∪ T = Iic 1 := by sorry

end NUMINAMATH_CALUDE_complement_S_union_T_equals_interval_l905_90521


namespace NUMINAMATH_CALUDE_two_color_draw_count_l905_90598

def total_balls : ℕ := 6
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def blue_balls : ℕ := 1
def draw_count : ℕ := 3

def ways_two_colors : ℕ := 13

theorem two_color_draw_count :
  ways_two_colors = (total_balls.choose draw_count) - 
    (red_balls * white_balls * blue_balls) - 
    (if white_balls ≥ draw_count then 1 else 0) :=
by sorry

end NUMINAMATH_CALUDE_two_color_draw_count_l905_90598


namespace NUMINAMATH_CALUDE_wheat_packets_fill_gunny_bag_l905_90546

/-- The number of pounds in one ton -/
def pounds_per_ton : ℕ := 2300

/-- The number of packets of wheat -/
def num_packets : ℕ := 1840

/-- The weight of each packet in pounds -/
def packet_weight_pounds : ℕ := 16

/-- The additional weight of each packet in ounces -/
def packet_weight_ounces : ℕ := 4

/-- The capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℕ := 13

/-- The number of ounces in one pound -/
def ounces_per_pound : ℕ := 16

theorem wheat_packets_fill_gunny_bag :
  ounces_per_pound = 16 :=
sorry

end NUMINAMATH_CALUDE_wheat_packets_fill_gunny_bag_l905_90546


namespace NUMINAMATH_CALUDE_coordinate_proof_l905_90527

/-- 
Given two points A(x₁, y₁) and B(x₂, y₂) in the first quadrant of a Cartesian coordinate system,
prove that under certain conditions, their coordinates are (1, 5) and (8, 9) respectively.
-/
theorem coordinate_proof (x₁ y₁ x₂ y₂ : ℕ) : 
  -- Both coordinates are positive integers
  0 < x₁ ∧ 0 < y₁ ∧ 0 < x₂ ∧ 0 < y₂ →
  -- Angle OA > 45°
  y₁ > x₁ →
  -- Angle OB < 45°
  x₂ > y₂ →
  -- Area difference condition
  x₂ * y₂ = x₁ * y₁ + 67 →
  -- Conclusion: coordinates are (1, 5) and (8, 9)
  x₁ = 1 ∧ y₁ = 5 ∧ x₂ = 8 ∧ y₂ = 9 := by
sorry

end NUMINAMATH_CALUDE_coordinate_proof_l905_90527


namespace NUMINAMATH_CALUDE_exist_three_equal_digit_sums_l905_90553

-- Define the sum of decimal digits function
def S (n : ℕ) : ℕ := sorry

-- State the theorem
theorem exist_three_equal_digit_sums :
  ∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 25 ∧
  S (a^6 + 2014) = S (b^6 + 2014) ∧ S (b^6 + 2014) = S (c^6 + 2014) := by sorry

end NUMINAMATH_CALUDE_exist_three_equal_digit_sums_l905_90553


namespace NUMINAMATH_CALUDE_cone_volume_l905_90523

/-- Given a cone with base circumference 2π and lateral area 2π, its volume is (√3 * π) / 3 -/
theorem cone_volume (r h l : ℝ) (h1 : 2 * π = 2 * π * r) (h2 : 2 * π = π * r * l) :
  (1 / 3) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l905_90523


namespace NUMINAMATH_CALUDE_modulo_equivalence_exists_unique_l905_90528

theorem modulo_equivalence_exists_unique : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_exists_unique_l905_90528


namespace NUMINAMATH_CALUDE_stating_mans_downstream_speed_l905_90592

/-- 
Given a man's upstream speed and the speed of a stream, 
this function calculates his downstream speed.
-/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  (upstream_speed + stream_speed) + stream_speed

/-- 
Theorem stating that given the specific conditions of the problem,
the man's downstream speed is 11 kmph.
-/
theorem mans_downstream_speed : 
  downstream_speed 8 1.5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_stating_mans_downstream_speed_l905_90592


namespace NUMINAMATH_CALUDE_handshakes_at_gathering_l905_90517

theorem handshakes_at_gathering (n : ℕ) (h : n = 6) : 
  n * (2 * n - 1) = 60 := by
  sorry

#check handshakes_at_gathering

end NUMINAMATH_CALUDE_handshakes_at_gathering_l905_90517


namespace NUMINAMATH_CALUDE_inequality_proof_l905_90585

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l905_90585


namespace NUMINAMATH_CALUDE_money_difference_l905_90500

/-- Proves that Hoseok has 170,000 won more than Min-young after they both earn additional money -/
theorem money_difference (initial_amount : ℕ) (minyoung_earnings hoseok_earnings : ℕ) :
  initial_amount = 1500000 →
  minyoung_earnings = 320000 →
  hoseok_earnings = 490000 →
  (initial_amount + hoseok_earnings) - (initial_amount + minyoung_earnings) = 170000 :=
by
  sorry

end NUMINAMATH_CALUDE_money_difference_l905_90500


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l905_90569

/-- Represents the lengths of three line segments -/
structure Triangle :=
  (a b c : ℝ)

/-- Checks if three line segments can form a triangle -/
def canFormTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- Theorem: The set of line segments 2cm, 3cm, 6cm cannot form a triangle -/
theorem cannot_form_triangle :
  ¬ canFormTriangle ⟨2, 3, 6⟩ :=
sorry

end NUMINAMATH_CALUDE_cannot_form_triangle_l905_90569


namespace NUMINAMATH_CALUDE_two_machine_completion_time_l905_90539

theorem two_machine_completion_time (t₁ t_combined : ℝ) (x : ℝ) 
  (h₁ : t₁ > 0) (h₂ : t_combined > 0) (h₃ : x > 0) 
  (h₄ : t₁ = 6) (h₅ : t_combined = 1.5) :
  (1 / t₁ + 1 / x = 1 / t_combined) ↔ 
  (1 / 6 + 1 / x = 1 / 1.5) :=
sorry

end NUMINAMATH_CALUDE_two_machine_completion_time_l905_90539


namespace NUMINAMATH_CALUDE_strawberry_cartons_correct_l905_90547

/-- Calculates the number of strawberry cartons in the cupboard given the total needed, blueberry cartons, and cartons bought. -/
def strawberry_cartons (total_needed : ℕ) (blueberry_cartons : ℕ) (cartons_bought : ℕ) : ℕ :=
  total_needed - (blueberry_cartons + cartons_bought)

/-- Theorem stating that the number of strawberry cartons is correct given the problem conditions. -/
theorem strawberry_cartons_correct :
  strawberry_cartons 42 7 33 = 2 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_cartons_correct_l905_90547


namespace NUMINAMATH_CALUDE_fraction_equality_implies_sum_l905_90506

theorem fraction_equality_implies_sum (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 64*x + 992) / (x^2 + 56*x - 3168)) →
  α + β = 82 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_sum_l905_90506


namespace NUMINAMATH_CALUDE_average_marker_cost_correct_l905_90549

def average_marker_cost (num_markers : ℕ) (marker_price : ℚ) (handling_fee : ℚ) (shipping_cost : ℚ) : ℕ :=
  let total_cost := marker_price + (num_markers : ℚ) * handling_fee + shipping_cost
  let total_cents := (total_cost * 100).floor
  let average_cents := (total_cents + (num_markers / 2)) / num_markers
  average_cents.toNat

theorem average_marker_cost_correct :
  average_marker_cost 300 45 0.1 8.5 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_marker_cost_correct_l905_90549


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l905_90577

theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) :
  a = (3, 2) →
  b.1 = x →
  b.2 = 4 →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = -8/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l905_90577


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l905_90567

-- Define the quadratic expression
def quadratic (k : ℝ) : ℝ := 8 * k^2 - 16 * k + 28

-- Define the completed square form
def completed_square (k a b c : ℝ) : ℝ := a * (k + b)^2 + c

-- Theorem statement
theorem quadratic_rewrite :
  ∃ (a b c : ℝ), 
    (∀ k, quadratic k = completed_square k a b c) ∧ 
    (c / b = -20) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l905_90567


namespace NUMINAMATH_CALUDE_unique_prime_seventh_power_l905_90522

theorem unique_prime_seventh_power (p : ℕ) : 
  Prime p ∧ ∃ q : ℕ, Prime q ∧ p + 25 = q^7 ↔ p = 103 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_seventh_power_l905_90522


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l905_90556

theorem sum_remainder_mod_seven (n : ℤ) : 
  ((7 - n) + (n + 3) + n^2) % 7 = (3 + n^2) % 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l905_90556


namespace NUMINAMATH_CALUDE_correct_train_sequence_l905_90557

-- Define the actions
inductive TrainAction
  | BuyTicket
  | WaitForTrain
  | CheckTicket
  | BoardTrain
  | RepairTrain

-- Define a sequence of actions
def ActionSequence := List TrainAction

-- Define the possible sequences
def sequenceA : ActionSequence := [TrainAction.BuyTicket, TrainAction.WaitForTrain, TrainAction.CheckTicket, TrainAction.BoardTrain]
def sequenceB : ActionSequence := [TrainAction.WaitForTrain, TrainAction.BuyTicket, TrainAction.BoardTrain, TrainAction.CheckTicket]
def sequenceC : ActionSequence := [TrainAction.BuyTicket, TrainAction.WaitForTrain, TrainAction.BoardTrain, TrainAction.CheckTicket]
def sequenceD : ActionSequence := [TrainAction.RepairTrain, TrainAction.BuyTicket, TrainAction.CheckTicket, TrainAction.BoardTrain]

-- Define the correct sequence
def correctSequence : ActionSequence := sequenceA

-- Theorem stating that sequenceA is the correct sequence
theorem correct_train_sequence : correctSequence = sequenceA := by
  sorry


end NUMINAMATH_CALUDE_correct_train_sequence_l905_90557


namespace NUMINAMATH_CALUDE_orthogonal_medians_theorem_l905_90599

/-- Given a triangle with side lengths a, b, c and medians ma, mb, mc,
    if ma is perpendicular to mb, then the medians form a right-angled triangle
    and the inequality 5(a^2 + b^2 - c^2) ≥ 8ab holds. -/
theorem orthogonal_medians_theorem (a b c ma mb mc : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ma : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_mb : mb^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (h_mc : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4)
  (h_perp : ma * mb = 0) : 
  ma^2 + mb^2 = mc^2 ∧ 5*(a^2 + b^2 - c^2) ≥ 8*a*b :=
sorry

end NUMINAMATH_CALUDE_orthogonal_medians_theorem_l905_90599


namespace NUMINAMATH_CALUDE_fibonacci_product_theorem_l905_90529

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sum of squares of divisors -/
def sum_of_squares_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => x * x)

/-- Main theorem -/
theorem fibonacci_product_theorem (N : ℕ) (h_pos : N > 0)
  (h_sum : sum_of_squares_of_divisors N = N * (N + 3)) :
  ∃ i j, N = fib i * fib j :=
sorry

end NUMINAMATH_CALUDE_fibonacci_product_theorem_l905_90529


namespace NUMINAMATH_CALUDE_henry_bicycle_improvement_l905_90501

/-- Henry's bicycle ride improvement --/
theorem henry_bicycle_improvement (initial_laps initial_time current_laps current_time : ℚ) 
  (h1 : initial_laps = 15)
  (h2 : initial_time = 45)
  (h3 : current_laps = 18)
  (h4 : current_time = 42) :
  (initial_time / initial_laps) - (current_time / current_laps) = 2/3 := by
  sorry

#eval (45 : ℚ) / 15 - (42 : ℚ) / 18

end NUMINAMATH_CALUDE_henry_bicycle_improvement_l905_90501


namespace NUMINAMATH_CALUDE_max_mutually_touching_spheres_l905_90511

/-- A configuration of mutually touching spheres -/
structure SphereTouchConfiguration where
  n : ℕ                           -- number of spheres
  touches_all_others : n > 1      -- each sphere touches all others
  no_triple_touch : n > 2 → True  -- no three spheres touch at the same point

/-- The maximum number of spheres in a valid SphereTouchConfiguration is 5 -/
theorem max_mutually_touching_spheres (config : SphereTouchConfiguration) : config.n ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_mutually_touching_spheres_l905_90511


namespace NUMINAMATH_CALUDE_inequality_proof_l905_90589

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  3/16 ≤ (a/(1+a))^2 + (b/(1+b))^2 + (c/(1+c))^2 ∧ (a/(1+a))^2 + (b/(1+b))^2 + (c/(1+c))^2 ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l905_90589


namespace NUMINAMATH_CALUDE_min_value_quadratic_l905_90583

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 2007 ∧ ∀ x, 3 * x^2 - 12 * x + 2023 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l905_90583


namespace NUMINAMATH_CALUDE_total_amount_paid_l905_90510

def grape_quantity : ℕ := 10
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 55

theorem total_amount_paid : 
  grape_quantity * grape_rate + mango_quantity * mango_rate = 1195 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l905_90510


namespace NUMINAMATH_CALUDE_min_horizontal_distance_l905_90525

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

-- Define the set of x-coordinates for points P
def P : Set ℝ := {x | f x = 5}

-- Define the set of x-coordinates for points Q
def Q : Set ℝ := {x | f x = -2}

-- State the theorem
theorem min_horizontal_distance :
  ∃ (p q : ℝ), p ∈ P ∧ q ∈ Q ∧
  ∀ (p' q' : ℝ), p' ∈ P → q' ∈ Q →
  |p - q| ≤ |p' - q'| ∧
  |p - q| = |Real.sqrt 6 - Real.sqrt 3| :=
sorry

end NUMINAMATH_CALUDE_min_horizontal_distance_l905_90525


namespace NUMINAMATH_CALUDE_linear_equation_m_value_l905_90533

theorem linear_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b, (4 - m) * x^(|m| - 3) - 16 = a * x + b) ∧ 
  (m - 4 ≠ 0) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_m_value_l905_90533


namespace NUMINAMATH_CALUDE_number_triples_satisfying_equation_l905_90580

theorem number_triples_satisfying_equation :
  ∀ (a b c : ℕ), a^(b+20) * (c-1) = c^(b+21) - 1 ↔ 
    ((a = 1 ∧ c = 0) ∨ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_number_triples_satisfying_equation_l905_90580


namespace NUMINAMATH_CALUDE_x_lt_2_necessary_not_sufficient_l905_90526

theorem x_lt_2_necessary_not_sufficient :
  ∃ (x : ℝ), x^2 - x - 2 < 0 → x < 2 ∧
  ∃ (y : ℝ), y < 2 ∧ ¬(y^2 - y - 2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_x_lt_2_necessary_not_sufficient_l905_90526


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l905_90512

theorem perpendicular_vectors_magnitude (a b : ℝ × ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⟂ b
  (a.1^2 + a.2^2 = 4) →          -- |a| = 2
  (b.1^2 + b.2^2 = 4) →          -- |b| = 2
  ((2*a.1 - b.1)^2 + (2*a.2 - b.2)^2 = 20) := by  -- |2a - b| = 2√5
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l905_90512
