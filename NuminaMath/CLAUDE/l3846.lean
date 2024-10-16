import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3846_384699

/-- An isosceles triangle with side lengths 4 and 9 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∀ a b c : ℝ,
      a = 4 ∧ b = 9 ∧ c = 9 →  -- Two sides are 9, one side is 4
      a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
      (a = b ∨ b = c ∨ c = a) →  -- Isosceles condition
      perimeter = a + b + c →  -- Perimeter definition
      perimeter = 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 22 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3846_384699


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l3846_384605

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + k ≠ 0) ↔ k > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l3846_384605


namespace NUMINAMATH_CALUDE_candy_bar_sales_l3846_384652

theorem candy_bar_sales (seth_sales : ℕ) (other_sales : ℕ) 
  (h1 : seth_sales = 3 * other_sales + 6) 
  (h2 : seth_sales = 78) : 
  other_sales = 24 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_sales_l3846_384652


namespace NUMINAMATH_CALUDE_codes_lost_with_no_leading_zeros_l3846_384660

/-- The number of digits in each code -/
def code_length : ℕ := 5

/-- The number of possible digits (0 to 9) -/
def digit_options : ℕ := 10

/-- The number of non-zero digits (1 to 9) -/
def non_zero_digits : ℕ := 9

/-- Calculates the total number of possible codes -/
def total_codes : ℕ := digit_options ^ code_length

/-- Calculates the number of codes without leading zeros -/
def codes_without_leading_zeros : ℕ := non_zero_digits * (digit_options ^ (code_length - 1))

/-- The theorem to be proved -/
theorem codes_lost_with_no_leading_zeros :
  total_codes - codes_without_leading_zeros = 10000 := by
  sorry


end NUMINAMATH_CALUDE_codes_lost_with_no_leading_zeros_l3846_384660


namespace NUMINAMATH_CALUDE_sum_of_roots_symmetric_function_l3846_384606

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The theorem stating that if g is symmetric about 3 and has exactly four distinct real roots,
    then the sum of these roots is 12 -/
theorem sum_of_roots_symmetric_function
  (g : ℝ → ℝ) 
  (h_sym : SymmetricAboutThree g)
  (h_roots : ∃! (s₁ s₂ s₃ s₄ : ℝ), s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₁ ≠ s₄ ∧ s₂ ≠ s₃ ∧ s₂ ≠ s₄ ∧ s₃ ≠ s₄ ∧ 
              g s₁ = 0 ∧ g s₂ = 0 ∧ g s₃ = 0 ∧ g s₄ = 0) :
  ∃ (s₁ s₂ s₃ s₄ : ℝ), g s₁ = 0 ∧ g s₂ = 0 ∧ g s₃ = 0 ∧ g s₄ = 0 ∧ s₁ + s₂ + s₃ + s₄ = 12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_symmetric_function_l3846_384606


namespace NUMINAMATH_CALUDE_fruit_garden_ratio_l3846_384659

/-- Given a garden with the specified conditions, prove the ratio of fruit section to whole garden --/
theorem fruit_garden_ratio 
  (total_area : ℝ) 
  (fruit_quarter : ℝ) 
  (h1 : total_area = 64) 
  (h2 : fruit_quarter = 8) : 
  (4 * fruit_quarter) / total_area = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_garden_ratio_l3846_384659


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l3846_384612

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -825 [ZMOD 8] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l3846_384612


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3846_384643

theorem rationalize_denominator :
  (3 : ℝ) / (Real.sqrt 50 + 2) = (15 * Real.sqrt 2 - 6) / 46 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3846_384643


namespace NUMINAMATH_CALUDE_inscribed_triangle_ratio_l3846_384656

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a triangle -/
structure Triangle where
  p : Point
  q : Point
  r : Point

theorem inscribed_triangle_ratio (a : ℝ) (b : ℝ) (c : ℝ) (e : Ellipse) (t : Triangle) :
  e.a = a ∧ e.b = b ∧
  c = (3/5) * a ∧
  t.q = Point.mk 0 b ∧
  t.p.y = t.r.y ∧
  t.p.x = -c ∧ t.r.x = c ∧
  (t.p.x^2 / a^2) + (t.p.y^2 / b^2) = 1 ∧
  (t.q.x^2 / a^2) + (t.q.y^2 / b^2) = 1 ∧
  (t.r.x^2 / a^2) + (t.r.y^2 / b^2) = 1 ∧
  2 * c = 0.6 * a →
  (Real.sqrt ((t.p.x - t.q.x)^2 + (t.p.y - t.q.y)^2)) / (t.r.x - t.p.x) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_ratio_l3846_384656


namespace NUMINAMATH_CALUDE_john_paid_1273_l3846_384677

/-- Calculates the amount John paid out of pocket for his purchases --/
def john_out_of_pocket (exchange_rate : ℝ) (computer_cost gaming_chair_cost accessories_cost : ℝ)
  (computer_discount gaming_chair_discount : ℝ) (sales_tax : ℝ)
  (playstation_value playstation_discount bicycle_price : ℝ) : ℝ :=
  let discounted_computer := computer_cost * (1 - computer_discount)
  let discounted_chair := gaming_chair_cost * (1 - gaming_chair_discount)
  let total_before_tax := discounted_computer + discounted_chair + accessories_cost
  let total_after_tax := total_before_tax * (1 + sales_tax)
  let playstation_sale := playstation_value * (1 - playstation_discount)
  let total_sales := playstation_sale + bicycle_price
  total_after_tax - total_sales

/-- Theorem stating that John paid $1273 out of pocket --/
theorem john_paid_1273 :
  john_out_of_pocket 100 1500 400 300 0.2 0.1 0.05 600 0.2 200 = 1273 := by
  sorry

end NUMINAMATH_CALUDE_john_paid_1273_l3846_384677


namespace NUMINAMATH_CALUDE_log_power_sum_l3846_384697

theorem log_power_sum (a b : ℝ) (h1 : a = Real.log 8) (h2 : b = Real.log 27) :
  (5 : ℝ) ^ (a / b) + 2 ^ (b / a) = 8 := by
  sorry

end NUMINAMATH_CALUDE_log_power_sum_l3846_384697


namespace NUMINAMATH_CALUDE_integer_solutions_of_manhattan_distance_equation_l3846_384663

def solution_set : Set (ℤ × ℤ) := {(2,2), (2,0), (3,1), (1,1)}

theorem integer_solutions_of_manhattan_distance_equation :
  {(x, y) : ℤ × ℤ | |x - 2| + |y - 1| = 1} = solution_set := by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_manhattan_distance_equation_l3846_384663


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3846_384675

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, (x + 1) * (x - 3) < 0 → x < 3) ∧
  (∃ x, x < 3 ∧ (x + 1) * (x - 3) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3846_384675


namespace NUMINAMATH_CALUDE_nh4oh_remaining_is_zero_l3846_384627

-- Define the molecules and their initial quantities
def NH4Cl : ℕ := 2
def NaOH : ℕ := 2
def H2SO4 : ℕ := 3
def KOH : ℕ := 4

-- Define the reactions
def reaction1 (nh4cl naoh : ℕ) : ℕ := min nh4cl naoh
def reaction2 (nh4cl h2so4 : ℕ) : ℕ := min (nh4cl / 2) h2so4 * 2
def reaction3 (naoh h2so4 : ℕ) : ℕ := min (naoh / 2) h2so4 * 2
def reaction4 (koh h2so4 : ℕ) : ℕ := min koh h2so4
def reaction5 (nh4oh koh : ℕ) : ℕ := min nh4oh koh

-- Theorem statement
theorem nh4oh_remaining_is_zero :
  let nh4oh_formed := reaction1 NH4Cl NaOH
  let h2so4_remaining := H2SO4 - reaction2 NH4Cl H2SO4
  let koh_remaining := KOH - reaction4 KOH h2so4_remaining
  nh4oh_formed - reaction5 nh4oh_formed koh_remaining = 0 := by sorry

end NUMINAMATH_CALUDE_nh4oh_remaining_is_zero_l3846_384627


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l3846_384683

theorem arithmetic_geometric_sequence_relation (a : ℕ → ℤ) (b : ℕ → ℝ) (d k m : ℕ) (q : ℝ) :
  (∀ n, a (n + 1) - a n = d) →
  (a k = k^2 + 2) →
  (a (2*k) = (k + 2)^2) →
  (k > 0) →
  (a 1 > 1) →
  (∀ n, b n = q^(n-1)) →
  (q > 0) →
  (∃ m : ℕ, m > 0 ∧ (3 * 2^2) / (3 * m^2) = 1 + q + q^2) →
  q = (Real.sqrt 13 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l3846_384683


namespace NUMINAMATH_CALUDE_top_book_cost_l3846_384666

/-- The cost of the "TOP" book -/
def top_cost : ℚ := 8

/-- The cost of the "ABC" book -/
def abc_cost : ℚ := 23

/-- The number of "TOP" books sold -/
def top_sold : ℕ := 13

/-- The number of "ABC" books sold -/
def abc_sold : ℕ := 4

/-- The difference in earnings between "TOP" and "ABC" books -/
def earnings_difference : ℚ := 12

theorem top_book_cost :
  top_cost * top_sold - abc_cost * abc_sold = earnings_difference :=
sorry

end NUMINAMATH_CALUDE_top_book_cost_l3846_384666


namespace NUMINAMATH_CALUDE_rosie_circles_count_l3846_384602

/-- Proves that given a circular track of 1/4 mile length, if person A runs 3 miles
    and person B runs at twice the speed of person A, then person B circles the track 24 times. -/
theorem rosie_circles_count (track_length : ℝ) (lou_distance : ℝ) (speed_ratio : ℝ) : 
  track_length = 1/4 →
  lou_distance = 3 →
  speed_ratio = 2 →
  (lou_distance * speed_ratio) / track_length = 24 := by
sorry

end NUMINAMATH_CALUDE_rosie_circles_count_l3846_384602


namespace NUMINAMATH_CALUDE_book_cost_problem_l3846_384694

theorem book_cost_problem (cost_loss : ℝ) (sell_price : ℝ) :
  cost_loss = 262.5 →
  sell_price = cost_loss * 0.85 →
  sell_price = (sell_price / 1.19) * 1.19 →
  cost_loss + (sell_price / 1.19) = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3846_384694


namespace NUMINAMATH_CALUDE_jones_elementary_population_l3846_384653

theorem jones_elementary_population :
  let total_students : ℕ := 225
  let boys_percentage : ℚ := 40 / 100
  let boys_count : ℕ := 90
  (boys_count : ℚ) / (total_students * boys_percentage) = 1 :=
by sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l3846_384653


namespace NUMINAMATH_CALUDE_ceiling_sum_equals_56_l3846_384604

theorem ceiling_sum_equals_56 :
  ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 30⌉^2 + ⌈Real.sqrt 300⌉ = 56 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_equals_56_l3846_384604


namespace NUMINAMATH_CALUDE_number_problem_l3846_384608

theorem number_problem (x y : ℝ) 
  (h1 : (40 / 100) * x = (30 / 100) * 50)
  (h2 : (60 / 100) * x = (45 / 100) * y) :
  x = 37.5 ∧ y = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3846_384608


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3846_384615

theorem min_value_expression (x : ℝ) (h : x > 0) :
  2 + 3 * x + 4 / x ≥ 2 + 4 * Real.sqrt 3 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) :
  2 + 3 * x + 4 / x = 2 + 4 * Real.sqrt 3 ↔ x = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3846_384615


namespace NUMINAMATH_CALUDE_selling_price_ratio_l3846_384617

theorem selling_price_ratio (c x y : ℝ) (hx : x = 0.80 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l3846_384617


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3846_384658

theorem solution_satisfies_system :
  let f (x y : ℝ) := x + y + 2 - 4*x*y
  ∀ (x y z : ℝ), 
    (f x y = 0 ∧ f y z = 0 ∧ f z x = 0) →
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3846_384658


namespace NUMINAMATH_CALUDE_third_term_is_nine_l3846_384670

/-- A sequence of 5 numbers with specific properties -/
def MagazineSequence (a : Fin 5 → ℕ) : Prop :=
  a 0 = 3 ∧ a 1 = 4 ∧ a 3 = 9 ∧ a 4 = 13 ∧
  ∀ i : Fin 3, (a (i + 1) - a i) - (a (i + 2) - a (i + 1)) = 
               (a (i + 2) - a (i + 1)) - (a (i + 3) - a (i + 2))

/-- The third term in the sequence is 9 -/
theorem third_term_is_nine (a : Fin 5 → ℕ) (h : MagazineSequence a) : a 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_nine_l3846_384670


namespace NUMINAMATH_CALUDE_dodecahedron_intersection_area_l3846_384633

/-- The area of a regular pentagon formed by intersecting a plane with a regular dodecahedron -/
theorem dodecahedron_intersection_area (s : ℝ) :
  let dodecahedron_side_length : ℝ := s
  let intersection_pentagon_side_length : ℝ := s / 2
  let intersection_pentagon_area : ℝ := (5 / 4) * (intersection_pentagon_side_length ^ 2) * ((Real.sqrt 5 + 1) / 2)
  intersection_pentagon_area = (5 * s^2 * (Real.sqrt 5 + 1)) / 16 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_intersection_area_l3846_384633


namespace NUMINAMATH_CALUDE_negation_of_positive_sum_l3846_384635

theorem negation_of_positive_sum (x y : ℝ) :
  (¬(x > 0 ∧ y > 0 → x + y > 0)) ↔ (x ≤ 0 ∨ y ≤ 0 → x + y ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_sum_l3846_384635


namespace NUMINAMATH_CALUDE_intersection_line_circle_l3846_384672

def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p | a * p.1 + b * p.2 = c}

theorem intersection_line_circle (a : ℝ) :
  let O : ℝ × ℝ := (0, 0)
  let C : Set (ℝ × ℝ) := Circle O 2
  let L : Set (ℝ × ℝ) := Line 1 1 a
  ∀ A B : ℝ × ℝ, A ∈ C ∩ L → B ∈ C ∩ L →
    ‖(A.1, A.2)‖ = ‖(A.1 + B.1, A.2 + B.2)‖ →
      a = 2 ∨ a = -2 :=
by
  sorry

#check intersection_line_circle

end NUMINAMATH_CALUDE_intersection_line_circle_l3846_384672


namespace NUMINAMATH_CALUDE_total_participants_is_260_l3846_384624

/-- Represents the voting scenario for a school disco date --/
structure VotingScenario where
  initial_oct22_percent : ℝ
  initial_oct29_percent : ℝ
  additional_oct22_votes : ℕ
  final_oct29_percent : ℝ

/-- Calculates the total number of participants in the voting --/
def total_participants (scenario : VotingScenario) : ℕ :=
  sorry

/-- Theorem stating that the total number of participants is 260 --/
theorem total_participants_is_260 (scenario : VotingScenario) 
  (h1 : scenario.initial_oct22_percent = 0.35)
  (h2 : scenario.initial_oct29_percent = 0.65)
  (h3 : scenario.additional_oct22_votes = 80)
  (h4 : scenario.final_oct29_percent = 0.45) :
  total_participants scenario = 260 := by
  sorry

end NUMINAMATH_CALUDE_total_participants_is_260_l3846_384624


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l3846_384673

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For a random variable ξ following a binomial distribution B(6, 1/3),
    the probability P(ξ = 2) is equal to 80/243 -/
theorem binomial_probability_two_successes :
  binomial_pmf 6 (1/3) 2 = 80/243 :=
sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l3846_384673


namespace NUMINAMATH_CALUDE_automotive_test_time_l3846_384621

/-- Proves that given a car driven the same distance three times at speeds of 4, 5, and 6 miles per hour respectively, and a total distance of 180 miles, the total time taken is 37 hours. -/
theorem automotive_test_time (total_distance : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_distance = 180 ∧ 
  speed1 = 4 ∧ 
  speed2 = 5 ∧ 
  speed3 = 6 → 
  (total_distance / (3 * speed1) + total_distance / (3 * speed2) + total_distance / (3 * speed3)) = 37 := by
  sorry

#check automotive_test_time

end NUMINAMATH_CALUDE_automotive_test_time_l3846_384621


namespace NUMINAMATH_CALUDE_warehouse_solution_l3846_384618

/-- Represents the problem of determining the number of warehouses on a straight road. -/
def WarehouseProblem (n : ℕ) : Prop :=
  -- n is odd
  ∃ k : ℕ, n = 2*k + 1 ∧
  -- Distance between warehouses is 1 km
  -- Each warehouse contains 8 tons of goods
  -- Truck capacity is 8 tons
  -- These conditions are implicit in the problem setup
  -- Optimal route distance is 300 km
  2 * k * (k + 1) - k = 300

/-- The solution to the warehouse problem is 25 warehouses. -/
theorem warehouse_solution : WarehouseProblem 25 := by
  sorry

#check warehouse_solution

end NUMINAMATH_CALUDE_warehouse_solution_l3846_384618


namespace NUMINAMATH_CALUDE_sandy_puppies_given_to_friends_l3846_384650

/-- Given the initial number of puppies and the number of puppies left,
    calculate the number of puppies given to friends. -/
def puppies_given_to_friends (initial_puppies left_puppies : ℕ) : ℕ :=
  initial_puppies - left_puppies

/-- Theorem stating that for Sandy's specific case, the number of puppies
    given to friends is 4. -/
theorem sandy_puppies_given_to_friends :
  puppies_given_to_friends 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandy_puppies_given_to_friends_l3846_384650


namespace NUMINAMATH_CALUDE_counterexample_exists_l3846_384638

theorem counterexample_exists : ∃ n : ℕ, 
  n > 1 ∧ ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 2)) ∧ ¬(Nat.Prime (n - 3)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3846_384638


namespace NUMINAMATH_CALUDE_motorcycle_trip_distance_l3846_384664

/-- Given a motorcycle trip from A to B to C with the following conditions:
  - The average speed for the entire trip is 25 miles per hour
  - The time from A to B is 3 times the time from B to C
  - The distance from B to C is half the distance from A to B
Prove that the distance from A to B is 100/3 miles -/
theorem motorcycle_trip_distance (average_speed : ℝ) (time_ratio : ℝ) (distance_ratio : ℝ) :
  average_speed = 25 →
  time_ratio = 3 →
  distance_ratio = 1/2 →
  ∃ (distance_AB : ℝ), distance_AB = 100/3 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_trip_distance_l3846_384664


namespace NUMINAMATH_CALUDE_book_ratio_is_three_to_one_l3846_384644

-- Define the number of books for each person
def elmo_books : ℕ := 24
def stu_books : ℕ := 4
def laura_books : ℕ := 2 * stu_books

-- Define the ratio of Elmo's books to Laura's books
def book_ratio : ℚ := elmo_books / laura_books

-- Theorem to prove
theorem book_ratio_is_three_to_one : book_ratio = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_is_three_to_one_l3846_384644


namespace NUMINAMATH_CALUDE_union_of_sets_l3846_384641

theorem union_of_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  M ∪ N = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3846_384641


namespace NUMINAMATH_CALUDE_rect_to_spherical_conversion_l3846_384626

/-- Conversion from rectangular to spherical coordinates -/
theorem rect_to_spherical_conversion :
  let x : ℝ := 2 * Real.sqrt 3
  let y : ℝ := 6
  let z : ℝ := -4
  let ρ : ℝ := 8
  let θ : ℝ := π / 3
  let φ : ℝ := 2 * π / 3
  (ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 0 ≤ φ ∧ φ ≤ π) →
  (x = ρ * Real.sin φ * Real.cos θ ∧
   y = ρ * Real.sin φ * Real.sin θ ∧
   z = ρ * Real.cos φ) :=
by sorry

end NUMINAMATH_CALUDE_rect_to_spherical_conversion_l3846_384626


namespace NUMINAMATH_CALUDE_parabola_two_distinct_roots_l3846_384630

/-- Given a real number m, the quadratic equation x^2 - (2m-1)x + (m^2 - m) = 0 has two distinct real roots. -/
theorem parabola_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁^2 - (2*m - 1)*x₁ + (m^2 - m) = 0 ∧
    x₂^2 - (2*m - 1)*x₂ + (m^2 - m) = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_two_distinct_roots_l3846_384630


namespace NUMINAMATH_CALUDE_all_roots_integer_iff_a_eq_50_l3846_384684

/-- The polynomial P(x) = x^3 - 2x^2 - 25x + a -/
def P (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 - 25*x + a

/-- A function that checks if a real number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The theorem stating that all roots of P(x) are integers iff a = 50 -/
theorem all_roots_integer_iff_a_eq_50 (a : ℝ) :
  (∀ x : ℝ, P a x = 0 → isInteger x) ↔ a = 50 := by sorry

end NUMINAMATH_CALUDE_all_roots_integer_iff_a_eq_50_l3846_384684


namespace NUMINAMATH_CALUDE_or_not_implies_right_l3846_384676

theorem or_not_implies_right (p q : Prop) : (p ∨ q) → ¬p → q := by
  sorry

end NUMINAMATH_CALUDE_or_not_implies_right_l3846_384676


namespace NUMINAMATH_CALUDE_expression_simplification_l3846_384616

theorem expression_simplification (x y : ℝ) 
  (h : y = Real.sqrt (x - 3) + Real.sqrt (6 - 2*x) + 2) : 
  Real.sqrt (2*x) * Real.sqrt (x/y) * (Real.sqrt (y/x) + Real.sqrt (1/y)) = 
    Real.sqrt 6 + (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3846_384616


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3846_384669

/-- Given two vectors a and b in ℝ³, prove that they are perpendicular if and only if x = 10/3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ × ℝ) :
  a = (2, -1, 3) → b = (-4, 2, x) → (a • b = 0 ↔ x = 10/3) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3846_384669


namespace NUMINAMATH_CALUDE_committee_selection_ways_l3846_384629

theorem committee_selection_ways : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l3846_384629


namespace NUMINAMATH_CALUDE_total_stamps_is_72_l3846_384632

/-- Calculates the total number of stamps needed for Valerie's mailing --/
def total_stamps : ℕ :=
  let thank_you_cards := 5
  let thank_you_stamps_per_card := 2
  let water_bill_stamps := 3
  let electric_bill_stamps := 2
  let internet_bill_stamps := 5
  let rebate_stamps_per_envelope := 2
  let job_app_stamps_per_envelope := 1
  let bill_types := 3
  let additional_rebates := 3

  let bill_stamps := water_bill_stamps + electric_bill_stamps + internet_bill_stamps
  let rebates := bill_types + additional_rebates
  let job_applications := 2 * rebates

  thank_you_cards * thank_you_stamps_per_card +
  bill_stamps +
  rebates * rebate_stamps_per_envelope +
  job_applications * job_app_stamps_per_envelope

theorem total_stamps_is_72 : total_stamps = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_stamps_is_72_l3846_384632


namespace NUMINAMATH_CALUDE_polynomial_factor_l3846_384686

theorem polynomial_factor (x y z : ℝ) :
  ∃ (q : ℝ → ℝ → ℝ → ℝ), 
    x^2 - y^2 - z^2 - 2*y*z + x - y - z + 2 = (x - y - z + 1) * q x y z := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l3846_384686


namespace NUMINAMATH_CALUDE_abs_equation_unique_solution_l3846_384674

theorem abs_equation_unique_solution :
  ∃! x : ℝ, |x - 9| = |x + 3| := by
sorry

end NUMINAMATH_CALUDE_abs_equation_unique_solution_l3846_384674


namespace NUMINAMATH_CALUDE_x_plus_y_values_l3846_384689

theorem x_plus_y_values (x y : ℝ) (hx : -x = 3) (hy : |y| = 5) : 
  x + y = -8 ∨ x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l3846_384689


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3846_384601

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h_arith : a 1 + 2 * a 2 = a 3) :
  (a 9 + a 10) / (a 9 + a 8) = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3846_384601


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l3846_384623

open Real

theorem trigonometric_equation_solutions (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    sin (5 * a) * cos x₁ - cos (x₁ + 4 * a) = 0 ∧
    sin (5 * a) * cos x₂ - cos (x₂ + 4 * a) = 0 ∧
    ¬ ∃ k : ℤ, x₁ - x₂ = π * (k : ℝ)) ↔
  ∃ t : ℤ, a = π * ((4 * t + 1 : ℤ) : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l3846_384623


namespace NUMINAMATH_CALUDE_radical_simplification_l3846_384678

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (42 * p) * Real.sqrt (7 * p) * Real.sqrt (14 * p) = 42 * p * Real.sqrt (7 * p) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l3846_384678


namespace NUMINAMATH_CALUDE_point_A_moves_to_vertex_3_l3846_384620

/-- Represents a vertex of a cube --/
structure Vertex where
  label : Nat
  onGreenFace : Bool
  onDistantWhiteFace : Bool
  onBottomRightWhiteFace : Bool

/-- Represents the rotation of a cube --/
def rotatedCube : List Vertex → List Vertex := sorry

/-- The initial position of point A --/
def pointA : Vertex := {
  label := 0,
  onGreenFace := true,
  onDistantWhiteFace := true,
  onBottomRightWhiteFace := true
}

/-- Theorem stating that point A moves to vertex 3 after rotation --/
theorem point_A_moves_to_vertex_3 (cube : List Vertex) :
  ∃ v ∈ rotatedCube cube,
    v.label = 3 ∧
    v.onGreenFace = true ∧
    v.onDistantWhiteFace = true ∧
    v.onBottomRightWhiteFace = true :=
  sorry

end NUMINAMATH_CALUDE_point_A_moves_to_vertex_3_l3846_384620


namespace NUMINAMATH_CALUDE_median_of_dataset2_with_X_l3846_384692

def dataset1 : List ℕ := [15, 9, 11, 7]
def dataset2 : List ℕ := [10, 11, 14, 8]

def mode (l : List ℕ) : ℕ := sorry
def median (l : List ℕ) : ℚ := sorry

theorem median_of_dataset2_with_X (X : ℕ) : 
  mode (X :: dataset1) = 11 → median (X :: dataset2) = 11 := by sorry

end NUMINAMATH_CALUDE_median_of_dataset2_with_X_l3846_384692


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3846_384637

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x + y ≥ 4) ∧
  (∃ x y : ℝ, x + y ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3846_384637


namespace NUMINAMATH_CALUDE_inequality_solution_count_l3846_384631

theorem inequality_solution_count : 
  ∃! (s : Finset ℤ), 
    (∀ n : ℤ, n ∈ s ↔ Real.sqrt (3*n - 1) ≤ Real.sqrt (5*n - 7) ∧ 
                       Real.sqrt (5*n - 7) < Real.sqrt (3*n + 8)) ∧ 
    s.card = 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l3846_384631


namespace NUMINAMATH_CALUDE_part1_part2_l3846_384646

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 6

-- Part 1
theorem part1 : ∀ x : ℝ, f 5 x < 0 ↔ -3 < x ∧ x < -2 := by sorry

-- Part 2
theorem part2 : ∀ a : ℝ, (∀ x : ℝ, f a x > 0) ↔ -2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3846_384646


namespace NUMINAMATH_CALUDE_product_of_sqrt_diff_eq_one_l3846_384667

theorem product_of_sqrt_diff_eq_one :
  let A := Real.sqrt 3008 + Real.sqrt 3009
  let B := -Real.sqrt 3008 - Real.sqrt 3009
  let C := Real.sqrt 3008 - Real.sqrt 3009
  let D := Real.sqrt 3009 - Real.sqrt 3008
  A * B * C * D = 1 := by
sorry

end NUMINAMATH_CALUDE_product_of_sqrt_diff_eq_one_l3846_384667


namespace NUMINAMATH_CALUDE_intersection_A_B_l3846_384611

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x * (x - 2) < 0}

def B : Set ℝ := {x | x - 1 > 0}

theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3846_384611


namespace NUMINAMATH_CALUDE_banana_cost_l3846_384679

/-- Given a rate of $6 for 4 pounds of bananas, prove that 20 pounds of bananas cost $30 -/
theorem banana_cost (rate : ℚ) (rate_pounds : ℚ) (desired_pounds : ℚ) : 
  rate = 6 → rate_pounds = 4 → desired_pounds = 20 → 
  (rate / rate_pounds) * desired_pounds = 30 := by
sorry

end NUMINAMATH_CALUDE_banana_cost_l3846_384679


namespace NUMINAMATH_CALUDE_root_implies_h_value_l3846_384622

theorem root_implies_h_value (h : ℝ) :
  ((-3 : ℝ)^3 + h * (-3) - 10 = 0) → h = -37/3 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_h_value_l3846_384622


namespace NUMINAMATH_CALUDE_line_l_equation_line_l_prime_equation_l3846_384661

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (-1, 2)

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the symmetry point
def sym_point : ℝ × ℝ := (1, -1)

-- Theorem for the equation of line l
theorem line_l_equation : 
  ∃ (m : ℝ), ∀ (x y : ℝ), 
    (l₁ x y ∧ l₂ x y → (x, y) = M) → 
    (∀ (a b : ℝ), perp_line a b → (y - M.2) = m * (x - M.1)) → 
    (x - 2 * y + 5 = 0) :=
sorry

-- Theorem for the equation of line l′
theorem line_l_prime_equation :
  ∀ (x y : ℝ),
    (∃ (x' y' : ℝ), l₁ x' y' ∧ 
      x' = 2 * sym_point.1 - x ∧ 
      y' = 2 * sym_point.2 - y) →
    (3 * x + 4 * y + 7 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_line_l_prime_equation_l3846_384661


namespace NUMINAMATH_CALUDE_integral_exp_abs_l3846_384654

theorem integral_exp_abs : ∫ x in (-2)..4, Real.exp (abs x) = Real.exp 4 + Real.exp (-2) - 2 := by sorry

end NUMINAMATH_CALUDE_integral_exp_abs_l3846_384654


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3846_384609

/-- 
Given three consecutive terms of an arithmetic sequence with common difference 6,
prove that if their sum is 342, then the terms are 108, 114, and 120.
-/
theorem arithmetic_sequence_sum (a b c : ℕ) : 
  (b = a + 6 ∧ c = b + 6) →  -- consecutive terms with common difference 6
  (a + b + c = 342) →        -- sum is 342
  (a = 108 ∧ b = 114 ∧ c = 120) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3846_384609


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3846_384603

/-- A function f: ℝ → ℝ is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = x^3 + 2x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The condition p: f(x) is monotonically increasing in (-∞, +∞) -/
def p (m : ℝ) : Prop := MonotonicallyIncreasing (f m)

/-- The condition q: m ≥ 8x / (x^2 + 4) holds for any x > 0 -/
def q (m : ℝ) : Prop := ∀ x > 0, m ≥ 8*x / (x^2 + 4)

/-- p is a necessary but not sufficient condition for q -/
theorem p_necessary_not_sufficient_for_q :
  (∀ m : ℝ, q m → p m) ∧ (∃ m : ℝ, p m ∧ ¬q m) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3846_384603


namespace NUMINAMATH_CALUDE_bahs_equal_to_500_yahs_l3846_384695

-- Define the conversion rates
def bah_to_rah : ℚ := 30 / 20
def rah_to_yah : ℚ := 25 / 10

-- Define the target number of yahs
def target_yahs : ℕ := 500

-- Theorem statement
theorem bahs_equal_to_500_yahs :
  ⌊(target_yahs : ℚ) / (rah_to_yah * bah_to_rah)⌋ = 133 := by
  sorry

end NUMINAMATH_CALUDE_bahs_equal_to_500_yahs_l3846_384695


namespace NUMINAMATH_CALUDE_annies_class_size_l3846_384651

theorem annies_class_size :
  ∀ (total_spent : ℚ) (candy_cost : ℚ) (candies_per_classmate : ℕ) (candies_left : ℕ),
    total_spent = 8 →
    candy_cost = 1/10 →
    candies_per_classmate = 2 →
    candies_left = 12 →
    (total_spent / candy_cost - candies_left) / candies_per_classmate = 34 := by
  sorry

end NUMINAMATH_CALUDE_annies_class_size_l3846_384651


namespace NUMINAMATH_CALUDE_sams_work_days_l3846_384668

theorem sams_work_days (total_days : ℕ) (daily_wage : ℤ) (daily_loss : ℤ) (total_earnings : ℤ) :
  total_days = 20 ∧ daily_wage = 60 ∧ daily_loss = 30 ∧ total_earnings = 660 →
  ∃ (days_not_worked : ℕ),
    days_not_worked = 6 ∧
    days_not_worked ≤ total_days ∧
    (total_days - days_not_worked) * daily_wage - days_not_worked * daily_loss = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_sams_work_days_l3846_384668


namespace NUMINAMATH_CALUDE_reflection_theorem_l3846_384648

/-- Original function -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- Reflection line -/
def reflection_line : ℝ := -2

/-- Resulting function after reflection -/
def g (x : ℝ) : ℝ := 2 * x + 9

/-- Theorem stating that g is the reflection of f across x = -2 -/
theorem reflection_theorem :
  ∀ x : ℝ, g (2 * reflection_line - x) = f x :=
sorry

end NUMINAMATH_CALUDE_reflection_theorem_l3846_384648


namespace NUMINAMATH_CALUDE_sequence_sum_l3846_384698

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

def geometric_sequence (b₁ : ℕ) (r : ℕ) : ℕ → ℕ
  | n => b₁ * r^(n - 1)

theorem sequence_sum (a₁ : ℕ) :
  let a := arithmetic_sequence a₁ 2
  let b := geometric_sequence 1 2
  a (b 2) + a (b 3) + a (b 4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3846_384698


namespace NUMINAMATH_CALUDE_maria_change_l3846_384613

/-- The change Maria receives when buying apples -/
theorem maria_change (num_apples : ℕ) (price_per_apple : ℚ) (paid_amount : ℚ) : 
  num_apples = 5 → 
  price_per_apple = 3/4 → 
  paid_amount = 10 → 
  paid_amount - (num_apples : ℚ) * price_per_apple = 25/4 := by
  sorry

#check maria_change

end NUMINAMATH_CALUDE_maria_change_l3846_384613


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3846_384642

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 5) = 20 → ∃ y z : ℝ, x^2 - 2*x - 35 = 0 ∧ y + z = 2 ∧ (x = y ∨ x = z) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3846_384642


namespace NUMINAMATH_CALUDE_probability_eight_heads_ten_flips_probability_eight_heads_proof_l3846_384687

/-- The probability of getting exactly 8 heads in 10 flips of a fair coin -/
theorem probability_eight_heads_ten_flips : ℚ :=
  45 / 1024

/-- The number of ways to choose 8 items from 10 -/
def choose_eight_from_ten : ℕ := 45

/-- The total number of possible outcomes when flipping a fair coin 10 times -/
def total_outcomes : ℕ := 2^10

/-- Proof that the probability of getting exactly 8 heads in 10 flips of a fair coin
    is equal to the number of ways to choose 8 from 10 divided by the total number of outcomes -/
theorem probability_eight_heads_proof :
  probability_eight_heads_ten_flips = choose_eight_from_ten / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_probability_eight_heads_ten_flips_probability_eight_heads_proof_l3846_384687


namespace NUMINAMATH_CALUDE_marys_max_take_home_pay_l3846_384662

/-- Calculates Mary's take-home pay after taxes and insurance premium -/
def marys_take_home_pay (max_hours : ℕ) (regular_rate : ℚ) (overtime_rates : List ℚ) 
  (social_security_rate : ℚ) (medicare_rate : ℚ) (insurance_premium : ℚ) : ℚ :=
  sorry

/-- Theorem stating Mary's take-home pay for maximum hours worked -/
theorem marys_max_take_home_pay : 
  marys_take_home_pay 70 8 [1.25, 1.5, 1.75, 2] (8/100) (2/100) 50 = 706 := by
  sorry

end NUMINAMATH_CALUDE_marys_max_take_home_pay_l3846_384662


namespace NUMINAMATH_CALUDE_fraction_equality_l3846_384688

theorem fraction_equality : (18 : ℚ) / (9 * 47 * 5) = 2 / 235 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3846_384688


namespace NUMINAMATH_CALUDE_spider_legs_count_l3846_384691

/-- The number of spiders in the room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- The total number of spider legs in the room -/
def total_legs : ℕ := num_spiders * legs_per_spider

theorem spider_legs_count : total_legs = 32 := by
  sorry

end NUMINAMATH_CALUDE_spider_legs_count_l3846_384691


namespace NUMINAMATH_CALUDE_shepherds_pie_pieces_l3846_384682

theorem shepherds_pie_pieces (customers_shepherds : ℕ) (customers_chicken : ℕ) (chicken_pieces : ℕ) (total_pies : ℕ) :
  customers_shepherds = 52 →
  customers_chicken = 80 →
  chicken_pieces = 5 →
  total_pies = 29 →
  ∃ (shepherds_pieces : ℕ), 
    shepherds_pieces = 4 ∧
    (customers_shepherds / shepherds_pieces : ℚ) + (customers_chicken / chicken_pieces : ℚ) = total_pies :=
by sorry

end NUMINAMATH_CALUDE_shepherds_pie_pieces_l3846_384682


namespace NUMINAMATH_CALUDE_sum_of_first_20_a_l3846_384647

def odd_number (n : ℕ) : ℕ := 2 * n - 1

def a (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ k => odd_number (n * (n - 1) + 1 + k))

theorem sum_of_first_20_a : Finset.sum (Finset.range 20) (λ i => a (i + 1)) = 44100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_20_a_l3846_384647


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3846_384685

/-- Given a triangle with side lengths 7, 24, and 25 units, its area is 84 square units. -/
theorem triangle_area : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c area =>
    a = 7 ∧ b = 24 ∧ c = 25 → area = 84

/-- The statement of the theorem -/
theorem triangle_area_proof : ∃ (area : ℝ), triangle_area 7 24 25 area :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3846_384685


namespace NUMINAMATH_CALUDE_max_reverse_sum_theorem_l3846_384657

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def reverse_number (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem max_reverse_sum_theorem (a b : ℕ) 
  (h1 : is_three_digit a) 
  (h2 : is_three_digit b) 
  (h3 : a % 10 ≠ 0) 
  (h4 : b % 10 ≠ 0) 
  (h5 : a + b = 1372) : 
  ∃ (max : ℕ), reverse_number a + reverse_number b ≤ max ∧ max = 1372 := by
  sorry

end NUMINAMATH_CALUDE_max_reverse_sum_theorem_l3846_384657


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_divisible_by_13_l3846_384681

-- Define a function to calculate the sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_number_with_digit_sum_divisible_by_13 (n : ℕ) :
  ∃ k : ℕ, k ∈ Finset.range 79 ∧ (sum_of_digits (n + k)) % 13 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_divisible_by_13_l3846_384681


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3846_384640

theorem polynomial_factorization (x y : ℝ) : 2*x^2 - x*y - 15*y^2 = (2*x - 5*y) * (x - 3*y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3846_384640


namespace NUMINAMATH_CALUDE_video_votes_theorem_l3846_384649

theorem video_votes_theorem (total_votes : ℕ) (score : ℤ) (like_percentage : ℚ) : 
  like_percentage = 3/4 ∧ 
  score = 120 ∧ 
  (like_percentage * total_votes : ℚ).num * 2 - total_votes = score → 
  total_votes = 240 := by
sorry

end NUMINAMATH_CALUDE_video_votes_theorem_l3846_384649


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3846_384628

theorem geometric_sequence_fourth_term :
  ∀ (a : ℕ → ℝ),
    (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
    a 1 = 6 →                            -- First term
    a 5 = 1458 →                         -- Last term
    a 4 = 162 :=                         -- Fourth term to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3846_384628


namespace NUMINAMATH_CALUDE_star_three_four_l3846_384693

/-- Custom binary operation ※ -/
def star (a b : ℝ) : ℝ := 2 * a + b

/-- Theorem stating that 3※4 = 10 -/
theorem star_three_four : star 3 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l3846_384693


namespace NUMINAMATH_CALUDE_rectangle_width_l3846_384610

/-- Given a rectangle with perimeter 150 cm and length 15 cm greater than width, prove the width is 30 cm. -/
theorem rectangle_width (w l : ℝ) (h1 : l = w + 15) (h2 : 2 * l + 2 * w = 150) : w = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3846_384610


namespace NUMINAMATH_CALUDE_remainder_b39_mod_125_l3846_384607

def reverse_concatenate (n : ℕ) : ℕ :=
  -- Definition of b_n
  sorry

theorem remainder_b39_mod_125 : reverse_concatenate 39 % 125 = 21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_b39_mod_125_l3846_384607


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l3846_384619

/-- Represents the problem of determining a fuel tank's capacity --/
theorem fuel_tank_capacity :
  ∀ (capacity : ℝ) 
    (ethanol_percent_A : ℝ) 
    (ethanol_percent_B : ℝ) 
    (total_ethanol : ℝ) 
    (fuel_A_volume : ℝ),
  ethanol_percent_A = 0.12 →
  ethanol_percent_B = 0.16 →
  total_ethanol = 30 →
  fuel_A_volume = 106 →
  ethanol_percent_A * fuel_A_volume + 
  ethanol_percent_B * (capacity - fuel_A_volume) = total_ethanol →
  capacity = 214 := by
sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l3846_384619


namespace NUMINAMATH_CALUDE_expression_simplification_l3846_384634

theorem expression_simplification :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 5^32 + 7^32 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3846_384634


namespace NUMINAMATH_CALUDE_count_three_digit_no_seven_nine_l3846_384665

/-- The count of digits available for the hundreds place in a three-digit number, excluding 7 and 9 -/
def hundreds_choices : ℕ := 7

/-- The count of digits available for the tens and units places, excluding 7 and 9 -/
def tens_units_choices : ℕ := 8

/-- The theorem stating the count of three-digit numbers without 7 or 9 -/
theorem count_three_digit_no_seven_nine :
  hundreds_choices * tens_units_choices * tens_units_choices = 448 := by
  sorry

end NUMINAMATH_CALUDE_count_three_digit_no_seven_nine_l3846_384665


namespace NUMINAMATH_CALUDE_min_both_mozart_bach_l3846_384600

theorem min_both_mozart_bach (total : ℕ) (mozart_fans : ℕ) (bach_fans : ℕ)
  (h1 : total = 150)
  (h2 : mozart_fans = 120)
  (h3 : bach_fans = 110)
  : ∃ (both : ℕ), both ≥ 80 ∧ 
    both ≤ mozart_fans ∧ 
    both ≤ bach_fans ∧ 
    ∀ (x : ℕ), x < both → 
      (mozart_fans - x) + (bach_fans - x) > total := by
  sorry

end NUMINAMATH_CALUDE_min_both_mozart_bach_l3846_384600


namespace NUMINAMATH_CALUDE_estimated_red_balls_l3846_384636

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 10

/-- Represents the number of draws -/
def total_draws : ℕ := 100

/-- Represents the number of white balls drawn -/
def white_draws : ℕ := 40

/-- Theorem: Given the conditions, the estimated number of red balls is 6 -/
theorem estimated_red_balls :
  total_balls * (total_draws - white_draws) / total_draws = 6 := by
  sorry

end NUMINAMATH_CALUDE_estimated_red_balls_l3846_384636


namespace NUMINAMATH_CALUDE_dividend_calculation_l3846_384645

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 16) 
  (h2 : quotient = 8) 
  (h3 : remainder = 4) : 
  divisor * quotient + remainder = 132 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3846_384645


namespace NUMINAMATH_CALUDE_quadratic_expression_values_l3846_384655

theorem quadratic_expression_values (m n : ℤ) 
  (h1 : |m| = 3) 
  (h2 : |n| = 2) 
  (h3 : m < n) : 
  m^2 + m*n + n^2 = 7 ∨ m^2 + m*n + n^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_values_l3846_384655


namespace NUMINAMATH_CALUDE_min_value_of_f_l3846_384614

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then |x + a| + |x - 1| else x^2 - a*x + 2

theorem min_value_of_f (a : ℝ) : 
  (∀ x, f a x ≥ a) ∧ (∃ x, f a x = a) ↔ a ∈ ({-2 - 2*Real.sqrt 3, 2} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3846_384614


namespace NUMINAMATH_CALUDE_parallel_implies_magnitude_perpendicular_implies_k_obtuse_angle_implies_k_range_l3846_384625

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (k : ℝ) : ℝ × ℝ := (-3, k)

-- Theorem 1
theorem parallel_implies_magnitude (k : ℝ) :
  (∃ (t : ℝ), a = t • (b k)) → ‖b k‖ = 3 * Real.sqrt 5 := by
  sorry

-- Theorem 2
theorem perpendicular_implies_k :
  (a • (a + 2 • (b (1/4))) = 0) → (1/4 : ℝ) = 1/4 := by
  sorry

-- Theorem 3
theorem obtuse_angle_implies_k_range (k : ℝ) :
  (a • (b k) < 0) → k < 3/2 ∧ k ≠ -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_implies_magnitude_perpendicular_implies_k_obtuse_angle_implies_k_range_l3846_384625


namespace NUMINAMATH_CALUDE_complement_implies_a_value_l3846_384696

def I (a : ℤ) : Set ℤ := {2, 4, a^2 - a - 3}
def A (a : ℤ) : Set ℤ := {4, 1 - a}

theorem complement_implies_a_value (a : ℤ) :
  (I a) \ (A a) = {-1} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complement_implies_a_value_l3846_384696


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3846_384690

theorem arithmetic_calculation : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3846_384690


namespace NUMINAMATH_CALUDE_odd_function_complete_expression_l3846_384671

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_positive (x : ℝ) : ℝ :=
  -x^2 + x + 1

theorem odd_function_complete_expression 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x, f x = if x > 0 then f_positive x
             else if x = 0 then 0
             else x^2 + x - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_complete_expression_l3846_384671


namespace NUMINAMATH_CALUDE_compare_quadratic_expressions_compare_fractions_l3846_384680

-- Problem 1
theorem compare_quadratic_expressions (x : ℝ) : 2 * x^2 - x > x^2 + x - 2 := by
  sorry

-- Problem 2
theorem compare_fractions (c a b : ℝ) (hc : c > a) (ha : a > b) (hb : b > 0) :
  a / (c - a) > b / (c - b) := by
  sorry

end NUMINAMATH_CALUDE_compare_quadratic_expressions_compare_fractions_l3846_384680


namespace NUMINAMATH_CALUDE_multiplication_result_l3846_384639

theorem multiplication_result : (300000 : ℕ) * 100000 = 30000000000 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l3846_384639
