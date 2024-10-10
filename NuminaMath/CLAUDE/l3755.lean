import Mathlib

namespace acoustics_class_male_count_l3755_375531

/-- The number of male students in the acoustics class -/
def male_students : ℕ := 120

/-- The number of female students in the acoustics class -/
def female_students : ℕ := 100

/-- The percentage of male students who are engineering students -/
def male_eng_percent : ℚ := 25 / 100

/-- The percentage of female students who are engineering students -/
def female_eng_percent : ℚ := 20 / 100

/-- The percentage of male engineering students who passed the final exam -/
def male_pass_percent : ℚ := 20 / 100

/-- The percentage of female engineering students who passed the final exam -/
def female_pass_percent : ℚ := 25 / 100

/-- The percentage of all engineering students who passed the exam -/
def total_pass_percent : ℚ := 22 / 100

theorem acoustics_class_male_count :
  male_students = 120 ∧
  (male_eng_percent * male_students * male_pass_percent +
   female_eng_percent * female_students * female_pass_percent) =
  total_pass_percent * (male_eng_percent * male_students + female_eng_percent * female_students) :=
by sorry

end acoustics_class_male_count_l3755_375531


namespace red_exhausted_first_l3755_375570

/-- Represents the number of marbles of each color in the bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability that red marbles are the first to be exhausted -/
def probability_red_exhausted (bag : MarbleBag) : ℚ :=
  sorry

/-- The theorem stating the probability of red marbles being exhausted first -/
theorem red_exhausted_first (bag : MarbleBag) 
  (h1 : bag.red = 3) 
  (h2 : bag.blue = 5) 
  (h3 : bag.green = 7) : 
  probability_red_exhausted bag = 21 / 40 := by
  sorry

end red_exhausted_first_l3755_375570


namespace total_glue_blobs_is_96_l3755_375517

/-- Represents a layer in the pyramid --/
structure Layer where
  size : Nat
  deriving Repr

/-- Calculates the number of internal glue blobs within a layer --/
def internalGlueBlobs (layer : Layer) : Nat :=
  2 * layer.size * (layer.size - 1)

/-- Calculates the number of glue blobs between two adjacent layers --/
def interlayerGlueBlobs (upper : Layer) (lower : Layer) : Nat :=
  upper.size * upper.size * 4

/-- The pyramid structure --/
def pyramid : List Layer := [
  { size := 4 },
  { size := 3 },
  { size := 2 },
  { size := 1 }
]

/-- Theorem: The total number of glue blobs in the pyramid is 96 --/
theorem total_glue_blobs_is_96 : 
  (pyramid.map internalGlueBlobs).sum + 
  (List.zipWith interlayerGlueBlobs pyramid.tail pyramid).sum = 96 := by
  sorry

#eval (pyramid.map internalGlueBlobs).sum + 
      (List.zipWith interlayerGlueBlobs pyramid.tail pyramid).sum

end total_glue_blobs_is_96_l3755_375517


namespace four_possible_values_for_D_l3755_375590

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem four_possible_values_for_D :
  ∀ (A B C D : ℕ),
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    A < 10 → B < 10 → C < 10 → D < 10 →
    is_even A →
    is_odd B →
    A + B = D →
    C + D = D →
    (∃ (S : Finset ℕ), S.card = 4 ∧ ∀ d, d ∈ S ↔ (∃ a b, is_even a ∧ is_odd b ∧ a + b = d ∧ a < 10 ∧ b < 10 ∧ a ≠ b)) :=
by sorry

end four_possible_values_for_D_l3755_375590


namespace additional_license_plates_l3755_375527

theorem additional_license_plates 
  (initial_first : Nat) 
  (initial_second : Nat) 
  (initial_third : Nat) 
  (added_letters : Nat) 
  (h1 : initial_first = 5) 
  (h2 : initial_second = 3) 
  (h3 : initial_third = 4) 
  (h4 : added_letters = 1) : 
  (initial_first + added_letters) * (initial_second + added_letters) * (initial_third + added_letters) - 
  (initial_first * initial_second * initial_third) = 60 := by
sorry

end additional_license_plates_l3755_375527


namespace hike_up_time_l3755_375547

/-- Proves that the time taken to hike up a hill is 1.8 hours given specific conditions -/
theorem hike_up_time (up_speed down_speed total_time : ℝ) 
  (h1 : up_speed = 4)
  (h2 : down_speed = 6)
  (h3 : total_time = 3) : 
  ∃ (t : ℝ), t * up_speed = (total_time - t) * down_speed ∧ t = 1.8 := by
  sorry

#check hike_up_time

end hike_up_time_l3755_375547


namespace percentage_both_correct_l3755_375558

theorem percentage_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) :
  p_first = 0.63 →
  p_second = 0.49 →
  p_neither = 0.20 →
  p_first + p_second - (1 - p_neither) = 0.32 := by
sorry

end percentage_both_correct_l3755_375558


namespace james_stickers_after_birthday_l3755_375507

/-- The number of stickers James had after his birthday -/
def total_stickers (initial : ℕ) (birthday : ℕ) : ℕ :=
  initial + birthday

/-- Theorem stating that James had 61 stickers after his birthday -/
theorem james_stickers_after_birthday :
  total_stickers 39 22 = 61 := by
  sorry

end james_stickers_after_birthday_l3755_375507


namespace arithmetic_progression_product_l3755_375526

theorem arithmetic_progression_product (a₁ a₂ a₃ a₄ d : ℕ) : 
  a₁ * a₂ * a₃ = 6 ∧ 
  a₁ * a₂ * a₃ * a₄ = 24 ∧ 
  a₂ = a₁ + d ∧ 
  a₃ = a₁ + 2 * d ∧ 
  a₄ = a₁ + 3 * d ↔ 
  a₁ = 1 ∧ a₂ = 2 ∧ a₃ = 3 ∧ a₄ = 4 := by
sorry

end arithmetic_progression_product_l3755_375526


namespace geometric_sequence_sum_l3755_375500

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 + a 8 = 2 →
  a 6 * a 7 = -8 →
  a 2 + a 11 = -7 := by
  sorry

end geometric_sequence_sum_l3755_375500


namespace x_plus_y_range_l3755_375580

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  y = 3 * (floor x) + 4 ∧
  y = 4 * (floor (x - 3)) + 7 ∧
  x ≠ ↑(floor x)

-- Theorem statement
theorem x_plus_y_range (x y : ℝ) :
  conditions x y → 40 < x + y ∧ x + y < 41 := by
  sorry

end x_plus_y_range_l3755_375580


namespace compound_interest_rate_l3755_375542

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r)^10 = 9000) 
  (h2 : P * (1 + r)^11 = 9990) : 
  r = 0.11 := by
  sorry

end compound_interest_rate_l3755_375542


namespace school_referendum_l3755_375577

theorem school_referendum (U A B : Finset Nat) (h1 : Finset.card U = 250)
  (h2 : Finset.card A = 190) (h3 : Finset.card B = 150)
  (h4 : Finset.card (U \ (A ∪ B)) = 40) :
  Finset.card (A ∩ B) = 130 := by
  sorry

end school_referendum_l3755_375577


namespace travel_time_ratio_l3755_375540

theorem travel_time_ratio : 
  let distance : ℝ := 252
  let original_time : ℝ := 6
  let new_speed : ℝ := 28
  let new_time : ℝ := distance / new_speed
  let original_speed : ℝ := distance / original_time
  new_time / original_time = 3 / 2 := by sorry

end travel_time_ratio_l3755_375540


namespace haley_marbles_division_l3755_375575

/-- Given a number of marbles and a number of boys, calculate the number of marbles each boy receives when divided equally. -/
def marblesPerBoy (totalMarbles : ℕ) (numBoys : ℕ) : ℕ :=
  totalMarbles / numBoys

/-- Theorem stating that when 35 marbles are divided equally among 5 boys, each boy receives 7 marbles. -/
theorem haley_marbles_division :
  marblesPerBoy 35 5 = 7 := by
  sorry

end haley_marbles_division_l3755_375575


namespace ratio_a_to_c_l3755_375569

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 7) :
  a / c = 105 / 16 := by
  sorry

end ratio_a_to_c_l3755_375569


namespace cube_sum_divisibility_l3755_375535

theorem cube_sum_divisibility (x y z t : ℤ) (h : x^3 + y^3 = 3*(z^3 + t^3)) :
  3 ∣ x ∧ 3 ∣ y := by sorry

end cube_sum_divisibility_l3755_375535


namespace servant_worked_nine_months_l3755_375574

/-- Represents the salary and employment duration of a servant --/
structure ServantSalary where
  yearly_cash : ℕ  -- Yearly cash salary in Rupees
  turban_price : ℕ  -- Price of the turban in Rupees
  leaving_cash : ℕ  -- Cash received when leaving in Rupees
  months_worked : ℚ  -- Number of months worked

/-- Calculates the number of months a servant worked based on the given salary structure --/
def calculate_months_worked (s : ServantSalary) : ℚ :=
  let total_yearly_salary : ℚ := s.yearly_cash + s.turban_price
  let monthly_salary : ℚ := total_yearly_salary / 12
  let total_received : ℚ := s.leaving_cash + s.turban_price
  total_received / monthly_salary

/-- Theorem stating that the servant worked for approximately 9 months --/
theorem servant_worked_nine_months (s : ServantSalary) 
  (h1 : s.yearly_cash = 90)
  (h2 : s.turban_price = 70)
  (h3 : s.leaving_cash = 50) :
  ∃ ε > 0, |calculate_months_worked s - 9| < ε := by
  sorry

#eval calculate_months_worked { yearly_cash := 90, turban_price := 70, leaving_cash := 50, months_worked := 0 }

end servant_worked_nine_months_l3755_375574


namespace computer_repair_cost_l3755_375587

theorem computer_repair_cost (phone_cost laptop_cost : ℕ) 
  (phone_repairs laptop_repairs computer_repairs : ℕ) (total_earnings : ℕ) :
  phone_cost = 11 →
  laptop_cost = 15 →
  phone_repairs = 5 →
  laptop_repairs = 2 →
  computer_repairs = 2 →
  total_earnings = 121 →
  ∃ (computer_cost : ℕ), 
    phone_cost * phone_repairs + laptop_cost * laptop_repairs + computer_cost * computer_repairs = total_earnings ∧
    computer_cost = 18 :=
by sorry

end computer_repair_cost_l3755_375587


namespace zachary_crunches_l3755_375599

/-- Given that David did 4 crunches and 13 fewer crunches than Zachary,
    prove that Zachary did 17 crunches. -/
theorem zachary_crunches (david_crunches : ℕ) (difference : ℕ) 
  (h1 : david_crunches = 4)
  (h2 : difference = 13) :
  david_crunches + difference = 17 := by
  sorry

end zachary_crunches_l3755_375599


namespace zoo_layout_problem_l3755_375543

/-- The number of tiger enclosures in a zoo -/
def tigerEnclosures : ℕ := sorry

/-- The number of zebra enclosures in the zoo -/
def zebraEnclosures : ℕ := 2 * tigerEnclosures

/-- The number of giraffe enclosures in the zoo -/
def giraffeEnclosures : ℕ := 3 * zebraEnclosures

/-- The number of tigers per tiger enclosure -/
def tigersPerEnclosure : ℕ := 4

/-- The number of zebras per zebra enclosure -/
def zebrasPerEnclosure : ℕ := 10

/-- The number of giraffes per giraffe enclosure -/
def giraffesPerEnclosure : ℕ := 2

/-- The total number of animals in the zoo -/
def totalAnimals : ℕ := 144

theorem zoo_layout_problem :
  tigerEnclosures * tigersPerEnclosure +
  zebraEnclosures * zebrasPerEnclosure +
  giraffeEnclosures * giraffesPerEnclosure = totalAnimals ∧
  tigerEnclosures = 4 := by sorry

end zoo_layout_problem_l3755_375543


namespace equation_solution_l3755_375541

theorem equation_solution :
  ∃ x : ℝ, (x - 6) ^ 4 = (1 / 16)⁻¹ ∧ x = 8 :=
by sorry

end equation_solution_l3755_375541


namespace prime_iff_binomial_divisible_l3755_375589

theorem prime_iff_binomial_divisible (n : ℕ) (h : n > 1) : 
  Nat.Prime n ↔ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → n ∣ Nat.choose n k := by
  sorry

end prime_iff_binomial_divisible_l3755_375589


namespace train_length_calculation_l3755_375555

/-- Proves that given a train and a platform of equal length, if the train crosses the platform
    in one minute at a speed of 144 km/hr, then the length of the train is 1200 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) 
    (h1 : train_length = platform_length)
    (h2 : train_length + platform_length = 144 * 1000 / 60) : 
    train_length = 1200 := by
  sorry

end train_length_calculation_l3755_375555


namespace trajectory_intersection_slope_ratio_l3755_375584

-- Define the curve E: y² = 2x
def E : Set (ℝ × ℝ) := {p | p.2^2 = 2 * p.1}

-- Define points S and Q
def S : ℝ × ℝ := (2, 0)
def Q : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem trajectory_intersection_slope_ratio 
  (k₁ : ℝ) 
  (A B C D : ℝ × ℝ) 
  (hA : A ∈ E) 
  (hB : B ∈ E) 
  (hC : C ∈ E) 
  (hD : D ∈ E) 
  (hAB : (B.2 - A.2) = k₁ * (B.1 - A.1)) 
  (hABS : (A.2 - S.2) = k₁ * (A.1 - S.1)) 
  (hAC : (C.2 - A.2) * (Q.1 - A.1) = (Q.2 - A.2) * (C.1 - A.1)) 
  (hBD : (D.2 - B.2) * (Q.1 - B.1) = (Q.2 - B.2) * (D.1 - B.1)) :
  ∃ (k₂ : ℝ), (D.2 - C.2) = k₂ * (D.1 - C.1) ∧ k₂ / k₁ = 2 := by
sorry

end trajectory_intersection_slope_ratio_l3755_375584


namespace adam_orchard_apples_l3755_375537

/-- Represents the number of apples Adam collected from his orchard -/
def total_apples (daily_apples : ℕ) (days : ℕ) (remaining_apples : ℕ) : ℕ :=
  daily_apples * days + remaining_apples

/-- Theorem stating the total number of apples Adam collected -/
theorem adam_orchard_apples :
  total_apples 4 30 230 = 350 := by
  sorry

end adam_orchard_apples_l3755_375537


namespace negative_squares_inequality_l3755_375562

theorem negative_squares_inequality (x b a : ℝ) 
  (h1 : x < b) (h2 : b < a) (h3 : a < 0) : x^2 > b*x ∧ b*x > b^2 := by
  sorry

end negative_squares_inequality_l3755_375562


namespace tan_150_degrees_l3755_375582

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l3755_375582


namespace cosine_of_inclination_angle_l3755_375501

/-- A line in 2D space represented by its parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The inclination angle of a line -/
def inclinationAngle (l : ParametricLine) : ℝ := sorry

/-- The given line with parametric equations x = -2 + 3t and y = 3 - 4t -/
def givenLine : ParametricLine := {
  x := λ t => -2 + 3*t,
  y := λ t => 3 - 4*t
}

/-- Theorem stating that the cosine of the inclination angle of the given line is -3/5 -/
theorem cosine_of_inclination_angle :
  Real.cos (inclinationAngle givenLine) = -3/5 := by sorry

end cosine_of_inclination_angle_l3755_375501


namespace parallelogram_area_main_theorem_l3755_375593

/-- Represents a parallelogram formed by vertices of smaller triangles inside an equilateral triangle -/
structure Parallelogram where
  m : ℕ  -- Length of one side in terms of unit triangles
  n : ℕ  -- Length of the other side in terms of unit triangles

/-- The main theorem statement -/
theorem parallelogram_area (p : Parallelogram) : 
  (p.m > 1 ∧ p.n > 1) →  -- Sides must be greater than 1 unit triangle
  (p.m + p.n = 6) →      -- Sum of sides is 6 (derived from the 46 triangles condition)
  (p.m * p.n = 8 ∨ p.m * p.n = 9) := by
  sorry

/-- The equilateral triangle ABC -/
def ABC : Set (ℝ × ℝ) := sorry

/-- The set of 400 smaller equilateral triangles -/
def small_triangles : Set (Set (ℝ × ℝ)) := sorry

/-- The condition that the parallelogram is formed by vertices of smaller triangles inside ABC -/
def parallelogram_in_triangle (p : Parallelogram) : Prop := sorry

/-- The condition that the parallelogram sides are parallel to ABC sides -/
def parallel_to_ABC_sides (p : Parallelogram) : Prop := sorry

/-- The condition that exactly 46 smaller triangles have at least one point in common with the parallelogram sides -/
def triangles_touching_sides (p : Parallelogram) : Prop := sorry

/-- The main theorem combining all conditions -/
theorem main_theorem (p : Parallelogram) :
  parallelogram_in_triangle p →
  parallel_to_ABC_sides p →
  triangles_touching_sides p →
  (p.m * p.n = 8 ∨ p.m * p.n = 9) := by
  sorry

end parallelogram_area_main_theorem_l3755_375593


namespace sqrt_3_times_sqrt_12_l3755_375581

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l3755_375581


namespace estimate_sqrt_expression_l3755_375533

theorem estimate_sqrt_expression :
  5 < Real.sqrt (1/3) * Real.sqrt 27 + Real.sqrt 7 ∧
  Real.sqrt (1/3) * Real.sqrt 27 + Real.sqrt 7 < 6 :=
by sorry

end estimate_sqrt_expression_l3755_375533


namespace no_obtuse_right_triangle_l3755_375528

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of triangles
def Triangle.isRight (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: An obtuse right triangle cannot exist
theorem no_obtuse_right_triangle :
  ∀ t : Triangle,
  (t.angle1 + t.angle2 + t.angle3 = 180) →
  ¬(t.isRight ∧ t.isObtuse) :=
by
  sorry


end no_obtuse_right_triangle_l3755_375528


namespace pizza_payment_difference_l3755_375564

/-- Represents the cost structure and consumption of a pizza --/
structure PizzaOrder where
  total_slices : Nat
  plain_cost : Int
  cheese_slices : Nat
  veggie_slices : Nat
  topping_cost : Int
  jerry_plain_slices : Nat

/-- Calculates the difference in payment between Jerry and Tom --/
def payment_difference (order : PizzaOrder) : Int :=
  let total_cost := order.plain_cost + 2 * order.topping_cost
  let slice_cost := total_cost / order.total_slices
  let jerry_slices := order.cheese_slices + order.veggie_slices + order.jerry_plain_slices
  let tom_slices := order.total_slices - jerry_slices
  slice_cost * (jerry_slices - tom_slices)

/-- Theorem stating the difference in payment between Jerry and Tom --/
theorem pizza_payment_difference :
  ∃ (order : PizzaOrder),
    order.total_slices = 12 ∧
    order.plain_cost = 12 ∧
    order.cheese_slices = 4 ∧
    order.veggie_slices = 4 ∧
    order.topping_cost = 3 ∧
    order.jerry_plain_slices = 2 ∧
    payment_difference order = 12 := by
  sorry

end pizza_payment_difference_l3755_375564


namespace midpoints_collinear_l3755_375560

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Two mutually perpendicular lines passing through a point -/
structure PerpendicularLines where
  origin : ℝ × ℝ
  direction1 : ℝ × ℝ
  direction2 : ℝ × ℝ
  perpendicular : direction1.1 * direction2.1 + direction1.2 * direction2.2 = 0

/-- Intersection points of lines with triangle sides -/
def intersectionPoints (t : Triangle) (l : PerpendicularLines) : List (ℝ × ℝ) := sorry

/-- Midpoints of segments -/
def midpoints (points : List (ℝ × ℝ)) : List (ℝ × ℝ) := sorry

/-- Check if points are collinear -/
def areCollinear (points : List (ℝ × ℝ)) : Prop := sorry

/-- Main theorem -/
theorem midpoints_collinear (t : Triangle) :
  let o := orthocenter t
  let l := PerpendicularLines.mk o (1, 0) (0, 1) (by simp)
  let intersections := intersectionPoints t l
  let mids := midpoints intersections
  areCollinear mids := by sorry

end midpoints_collinear_l3755_375560


namespace no_solution_equation_l3755_375524

theorem no_solution_equation : ¬∃ (x : ℝ), x - 9 / (x - 5) = 5 - 9 / (x - 5) := by
  sorry

end no_solution_equation_l3755_375524


namespace nine_sided_polygon_diagonals_l3755_375508

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular nine-sided polygon contains 36 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 36 := by sorry

end nine_sided_polygon_diagonals_l3755_375508


namespace ninas_pet_insects_eyes_l3755_375553

/-- The total number of eyes among Nina's pet insects -/
def total_eyes (num_spiders num_ants spider_eyes ant_eyes : ℕ) : ℕ :=
  num_spiders * spider_eyes + num_ants * ant_eyes

/-- Theorem stating that the total number of eyes among Nina's pet insects is 124 -/
theorem ninas_pet_insects_eyes :
  total_eyes 3 50 8 2 = 124 := by
  sorry

end ninas_pet_insects_eyes_l3755_375553


namespace modified_short_bingo_first_column_l3755_375515

/-- The number of elements in the set from which we select numbers -/
def n : ℕ := 15

/-- The number of elements we select -/
def k : ℕ := 5

/-- The number of ways to select k distinct numbers from a set of n numbers, where order matters -/
def permutations (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

theorem modified_short_bingo_first_column : permutations n k = 360360 := by
  sorry

end modified_short_bingo_first_column_l3755_375515


namespace function_properties_l3755_375523

def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - m

theorem function_properties (m : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f m y ≤ f m x ∧ f m x = 0) →
  (m = 0 ∨ m = 4) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, ∀ y ∈ Set.Icc (-1 : ℝ) 0, x ≤ y → f m x ≥ f m y) →
  (m ≤ -2) ∧
  (Set.range (f m) = Set.Icc 2 3) ↔ m = 6 :=
by sorry

end function_properties_l3755_375523


namespace intersection_A_complement_B_l3755_375521

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end intersection_A_complement_B_l3755_375521


namespace not_p_or_not_q_false_implies_p_and_q_and_p_or_q_true_l3755_375548

theorem not_p_or_not_q_false_implies_p_and_q_and_p_or_q_true (p q : Prop) :
  (¬(¬p ∨ ¬q)) → ((p ∧ q) ∧ (p ∨ q)) := by
  sorry

end not_p_or_not_q_false_implies_p_and_q_and_p_or_q_true_l3755_375548


namespace equation_solution_l3755_375529

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 3 ∧ x₂ = 3/5 ∧ 
  ∀ (x : ℝ), (x - 3)^2 + 4*x*(x - 3) = 0 ↔ (x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l3755_375529


namespace largest_k_value_l3755_375518

/-- A function that splits the whole numbers from 1 to 2k into two groups -/
def split_numbers (k : ℕ) : (Fin (2 * k) → Bool) := sorry

/-- A predicate that checks if two numbers share more than two distinct prime factors -/
def share_more_than_two_prime_factors (a b : ℕ) : Prop := sorry

/-- The main theorem stating that 44 is the largest possible value of k -/
theorem largest_k_value : 
  ∀ k : ℕ, k > 44 → 
  ¬∃ (f : Fin (2 * k) → Bool), 
    (∀ i j : Fin (2 * k), i.val < j.val ∧ f i = f j → 
      ¬share_more_than_two_prime_factors (i.val + 1) (j.val + 1)) ∧
    (Fintype.card {i : Fin (2 * k) | f i = true} = k) :=
sorry

end largest_k_value_l3755_375518


namespace quadratic_equation_solution_l3755_375502

theorem quadratic_equation_solution (a : ℝ) : 
  (1 : ℝ)^2 + a*(1 : ℝ) + 1 = 0 → a = -2 := by
  sorry

end quadratic_equation_solution_l3755_375502


namespace inequality_proof_l3755_375565

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_condition : a + b + c < 2) : 
  Real.sqrt (a^2 + b*c) + Real.sqrt (b^2 + c*a) + Real.sqrt (c^2 + a*b) < 3 := by
  sorry

end inequality_proof_l3755_375565


namespace polynomial_factorization_l3755_375586

theorem polynomial_factorization (x : ℝ) : 
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) := by
  sorry

end polynomial_factorization_l3755_375586


namespace trapezoid_segment_length_l3755_375561

/-- Given a trapezoid ABCD, proves that if the ratio of the areas of triangles ABC and ADC is 5:2,
    and the sum of AB and CD is 280, then AB equals 200. -/
theorem trapezoid_segment_length (A B C D : Point) (h : ℝ) :
  let triangle_ABC := (1/2) * AB * h
  let triangle_ADC := (1/2) * CD * h
  triangle_ABC / triangle_ADC = 5/2 →
  AB + CD = 280 →
  AB = 200 :=
by
  sorry


end trapezoid_segment_length_l3755_375561


namespace no_x_squared_term_l3755_375511

theorem no_x_squared_term (a : ℚ) : 
  (∀ x, (-9 * x^3 + (-6*a - 4) * x^2 - 3*x) = (-9 * x^3 - 3*x)) ↔ a = -2/3 :=
by sorry

end no_x_squared_term_l3755_375511


namespace expression_value_l3755_375568

theorem expression_value (x : ℝ) (h : x^2 + 2*x = 2) :
  (x - 1)^2 + (x + 3)*(x - 3) - (x - 3)*(x - 1) = -9 := by
  sorry

end expression_value_l3755_375568


namespace calculate_expression_l3755_375512

theorem calculate_expression : (10^10 / (2 * 10^6)) * 3 = 15000 := by
  sorry

end calculate_expression_l3755_375512


namespace ratio_of_numbers_l3755_375503

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : a + b = 7 * (a - b)) (h5 : a^2 + b^2 = 85) : a / b = 4 / 3 := by
  sorry

end ratio_of_numbers_l3755_375503


namespace cubic_division_theorem_l3755_375520

theorem cubic_division_theorem (c d : ℝ) (hc : c = 7) (hd : d = 3) :
  (c^3 + d^3) / (c^2 - c*d + d^2) = 10 := by
  sorry

end cubic_division_theorem_l3755_375520


namespace triangle_cosine_relation_l3755_375559

theorem triangle_cosine_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (Real.cos (B-C) * Real.cos A + Real.cos (2*A) = 1 + Real.cos A * Real.cos (B+C)) →
  ((B = C → Real.cos A = 2/3) ∧ (b^2 + c^2) / a^2 = 3) := by
sorry

end triangle_cosine_relation_l3755_375559


namespace square_points_sum_l3755_375522

/-- Square with side length 1000 -/
structure Square :=
  (side : ℝ)
  (is_1000 : side = 1000)

/-- Points on the side of the square -/
structure PointOnSide (S : Square) :=
  (pos : ℝ)
  (on_side : 0 ≤ pos ∧ pos ≤ S.side)

/-- Condition that E is between A and F -/
def between (A E F : ℝ) : Prop := A ≤ E ∧ E ≤ F

/-- Angle in degrees -/
def angle (θ : ℝ) := 0 ≤ θ ∧ θ < 360

/-- Distance between two points on a line -/
def distance (x y : ℝ) := |x - y|

/-- Representation of BF as p + q√r -/
structure IrrationalForm :=
  (p q r : ℕ)
  (r_not_square : ∀ (n : ℕ), n > 1 → r % (n^2) ≠ 0)

theorem square_points_sum (S : Square) 
  (E F : PointOnSide S)
  (AE_less_BF : E.pos < S.side - F.pos)
  (E_between_A_F : between 0 E.pos F.pos)
  (angle_EOF : angle 30)
  (EF_length : distance E.pos F.pos = 500)
  (BF_form : IrrationalForm)
  (BF_value : S.side - F.pos = BF_form.p + BF_form.q * Real.sqrt BF_form.r) :
  BF_form.p + BF_form.q + BF_form.r = 253 := by
  sorry

end square_points_sum_l3755_375522


namespace linear_function_properties_l3755_375506

/-- A linear function passing through two given points -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem linear_function_properties :
  ∀ (k b : ℝ), k ≠ 0 →
  linear_function k b 2 = -3 →
  linear_function k b (-4) = 0 →
  (k = -1/2 ∧ b = -2) ∧
  (∀ (x m : ℝ), x > -2 → -x + m < linear_function k b x → m ≤ -3) :=
by sorry

end linear_function_properties_l3755_375506


namespace second_half_speed_l3755_375549

/-- Given a trip with the following properties:
  * Total distance is 60 km
  * First half of the trip (30 km) is traveled at 48 km/h
  * Average speed of the entire trip is 32 km/h
  Then the speed of the second half of the trip is 24 km/h -/
theorem second_half_speed (total_distance : ℝ) (first_half_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 60 →
  first_half_distance = 30 →
  first_half_speed = 48 →
  average_speed = 32 →
  let second_half_distance := total_distance - first_half_distance
  let total_time := total_distance / average_speed
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  let second_half_speed := second_half_distance / second_half_time
  second_half_speed = 24 := by
  sorry

end second_half_speed_l3755_375549


namespace fraction_power_equality_l3755_375591

theorem fraction_power_equality : (81000 ^ 5 : ℕ) / (9000 ^ 5 : ℕ) = 59049 := by
  sorry

end fraction_power_equality_l3755_375591


namespace conic_eccentricity_l3755_375525

/-- The eccentricity of a conic section x + y^2/m = 1, where m is the geometric mean of 2 and 8 -/
theorem conic_eccentricity (m : ℝ) : 
  (m^2 = 2 * 8) →  -- m is the geometric mean of 2 and 8
  (∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧ 
    ∃ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
      ((x + y^2/m = 1) → (e = c/a ∧ (a^2 = b^2 + c^2 ∨ a^2 + b^2 = c^2)))) :=
by sorry

end conic_eccentricity_l3755_375525


namespace min_garden_cost_l3755_375544

/-- Represents the dimensions of a rectangular region -/
structure Region where
  length : ℝ
  width : ℝ

/-- Represents a type of flower with its cost -/
structure Flower where
  name : String
  cost : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Calculates the cost of planting a flower in a region -/
def plantingCost (f : Flower) (r : Region) : ℝ := f.cost * area r

/-- The main theorem stating the minimum cost of the garden -/
theorem min_garden_cost (regions : List Region) (flowers : List Flower) : 
  regions.length = 5 →
  flowers.length = 5 →
  regions = [
    ⟨5, 2⟩, 
    ⟨7, 3⟩, 
    ⟨5, 5⟩, 
    ⟨2, 4⟩, 
    ⟨5, 4⟩
  ] →
  flowers = [
    ⟨"Marigold", 1⟩,
    ⟨"Sunflower", 1.75⟩,
    ⟨"Tulip", 1.25⟩,
    ⟨"Orchid", 2.75⟩,
    ⟨"Iris", 3.25⟩
  ] →
  ∃ (assignment : List (Flower × Region)), 
    assignment.length = 5 ∧ 
    (∀ f r, (f, r) ∈ assignment → f ∈ flowers ∧ r ∈ regions) ∧
    (∀ f, f ∈ flowers → ∃! r, (f, r) ∈ assignment) ∧
    (∀ r, r ∈ regions → ∃! f, (f, r) ∈ assignment) ∧
    (assignment.map (λ (f, r) => plantingCost f r)).sum = 140.75 ∧
    ∀ (other_assignment : List (Flower × Region)),
      other_assignment.length = 5 →
      (∀ f r, (f, r) ∈ other_assignment → f ∈ flowers ∧ r ∈ regions) →
      (∀ f, f ∈ flowers → ∃! r, (f, r) ∈ other_assignment) →
      (∀ r, r ∈ regions → ∃! f, (f, r) ∈ other_assignment) →
      (other_assignment.map (λ (f, r) => plantingCost f r)).sum ≥ 140.75 :=
by sorry

end min_garden_cost_l3755_375544


namespace profit_percentage_l3755_375532

theorem profit_percentage (C P : ℝ) (h : (2/3) * P = 0.95 * C) : 
  (P - C) / C * 100 = 42.5 := by
sorry

end profit_percentage_l3755_375532


namespace exists_h_for_phi_l3755_375596

-- Define the types for our functions
def φ : ℝ → ℝ → ℝ → ℝ := sorry
def f : ℝ → ℝ → ℝ := sorry
def g : ℝ → ℝ → ℝ := sorry

-- State the theorem
theorem exists_h_for_phi (hf : ∀ x y z, φ x y z = f (x + y) z)
                         (hg : ∀ x y z, φ x y z = g x (y + z)) :
  ∃ h : ℝ → ℝ, ∀ x y z, φ x y z = h (x + y + z) := by sorry

end exists_h_for_phi_l3755_375596


namespace inequality_proof_l3755_375557

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : a * b + b * c + c * a = a + b + c) : 
  a^2 + b^2 + c^2 + 2*a*b*c ≥ 5 := by
  sorry

end inequality_proof_l3755_375557


namespace range_of_a_l3755_375578

theorem range_of_a (a : ℝ) : 
  let A := {x : ℝ | x > a}
  let B := {x : ℝ | x > 6}
  A ⊆ B ↔ a ≥ 6 := by sorry

end range_of_a_l3755_375578


namespace probability_both_selected_l3755_375513

/-- The probability of both Ram and Ravi being selected in an exam -/
theorem probability_both_selected (prob_ram prob_ravi : ℚ) 
  (h_ram : prob_ram = 4 / 7)
  (h_ravi : prob_ravi = 1 / 5) :
  prob_ram * prob_ravi = 4 / 35 := by
  sorry

end probability_both_selected_l3755_375513


namespace squirrel_count_l3755_375597

theorem squirrel_count (first_count : ℕ) (second_count : ℕ) : 
  first_count = 12 →
  second_count = first_count + first_count / 3 →
  first_count + second_count = 28 := by
sorry

end squirrel_count_l3755_375597


namespace opposite_absolute_values_l3755_375509

theorem opposite_absolute_values (a b : ℝ) :
  (|a - 1| + |b - 2| = 0) → (a + b = 3) := by
  sorry

end opposite_absolute_values_l3755_375509


namespace solve_equation_l3755_375579

theorem solve_equation : ∃! x : ℝ, 3 * x - 2 * (10 - x) = 5 ∧ x = 5 := by sorry

end solve_equation_l3755_375579


namespace quadratic_form_minimum_l3755_375514

theorem quadratic_form_minimum (x y : ℝ) : 3*x^2 + 2*x*y + y^2 - 6*x + 4*y + 9 ≥ 0 ∧
  ∃ (x₀ y₀ : ℝ), 3*x₀^2 + 2*x₀*y₀ + y₀^2 - 6*x₀ + 4*y₀ + 9 = 0 := by
  sorry

end quadratic_form_minimum_l3755_375514


namespace cos_triple_angle_l3755_375536

theorem cos_triple_angle (α : ℝ) : Real.cos (3 * α) = 4 * (Real.cos α)^3 - 3 * Real.cos α := by
  sorry

end cos_triple_angle_l3755_375536


namespace rose_puzzle_l3755_375538

theorem rose_puzzle : ∃! n : ℕ, 
  300 ≤ n ∧ n ≤ 400 ∧ 
  n % 21 = 13 ∧ 
  n % 15 = 7 ∧ 
  n = 307 := by sorry

end rose_puzzle_l3755_375538


namespace x_minus_y_equals_negative_three_l3755_375534

theorem x_minus_y_equals_negative_three
  (eq1 : 2020 * x + 2024 * y = 2028)
  (eq2 : 2022 * x + 2026 * y = 2030)
  : x - y = -3 := by
  sorry

end x_minus_y_equals_negative_three_l3755_375534


namespace loan_shark_fees_l3755_375510

/-- Calculates the total fees for a loan with a doubling weekly rate -/
def totalFees (loanAmount : ℝ) (initialRate : ℝ) (weeks : ℕ) : ℝ :=
  let weeklyFees := fun w => loanAmount * initialRate * (2 ^ w)
  (Finset.range weeks).sum weeklyFees

/-- Theorem stating that the total fees for a $100 loan at 5% initial rate for 2 weeks is $15 -/
theorem loan_shark_fees : totalFees 100 0.05 2 = 15 := by
  sorry

end loan_shark_fees_l3755_375510


namespace no_convex_polygon_partition_into_non_convex_quadrilaterals_l3755_375563

/-- A polygon is a closed planar figure bounded by straight line segments. -/
structure Polygon where
  vertices : Set (ℝ × ℝ)
  is_closed : Bool
  is_planar : Bool

/-- A polygon is convex if all its interior angles are less than or equal to 180 degrees. -/
def is_convex (p : Polygon) : Prop :=
  sorry

/-- A quadrilateral is a polygon with exactly four sides. -/
def is_quadrilateral (p : Polygon) : Prop :=
  sorry

/-- A quadrilateral is non-convex if at least one of its interior angles is greater than 180 degrees. -/
def is_non_convex_quadrilateral (q : Polygon) : Prop :=
  is_quadrilateral q ∧ ¬(is_convex q)

/-- A partition of a polygon is a set of smaller polygons that completely cover the original polygon without overlapping. -/
def is_partition (p : Polygon) (parts : Set Polygon) : Prop :=
  sorry

/-- The main theorem: It is impossible to partition a convex polygon into non-convex quadrilaterals. -/
theorem no_convex_polygon_partition_into_non_convex_quadrilaterals :
  ∀ (p : Polygon) (parts : Set Polygon),
    is_convex p →
    is_partition p parts →
    (∀ q ∈ parts, is_non_convex_quadrilateral q) →
    False :=
  sorry

end no_convex_polygon_partition_into_non_convex_quadrilaterals_l3755_375563


namespace tempo_insurance_premium_l3755_375516

/-- Calculate the premium amount for a tempo insurance --/
theorem tempo_insurance_premium 
  (original_value : ℝ) 
  (insurance_extent : ℝ) 
  (premium_rate : ℝ) 
  (h1 : original_value = 14000)
  (h2 : insurance_extent = 5/7)
  (h3 : premium_rate = 3/100) : 
  original_value * insurance_extent * premium_rate = 300 := by
  sorry

end tempo_insurance_premium_l3755_375516


namespace quadratic_min_max_l3755_375545

/-- The quadratic function f(x) = 2x^2 - 8x + 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 3

theorem quadratic_min_max :
  (∀ x : ℝ, f x ≥ -5) ∧
  (f 2 = -5) ∧
  (∀ M : ℝ, ∃ x : ℝ, f x > M) :=
by sorry

end quadratic_min_max_l3755_375545


namespace goods_train_speed_calculation_l3755_375539

/-- The speed of the man's train in km/h -/
def man_train_speed : ℝ := 60

/-- The length of the goods train in meters -/
def goods_train_length : ℝ := 280

/-- The time it takes for the goods train to pass the man in seconds -/
def passing_time : ℝ := 9

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 52

theorem goods_train_speed_calculation :
  (man_train_speed + goods_train_speed) * passing_time / 3600 = goods_train_length / 1000 :=
by sorry

end goods_train_speed_calculation_l3755_375539


namespace students_in_both_clubs_l3755_375576

theorem students_in_both_clubs 
  (total_students : ℕ) 
  (drama_students : ℕ) 
  (science_students : ℕ) 
  (students_in_either : ℕ) 
  (h1 : total_students = 320)
  (h2 : drama_students = 90)
  (h3 : science_students = 140)
  (h4 : students_in_either = 200) :
  drama_students + science_students - students_in_either = 30 :=
by sorry

end students_in_both_clubs_l3755_375576


namespace inequality_implication_l3755_375573

theorem inequality_implication (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

end inequality_implication_l3755_375573


namespace prob_even_sum_is_seven_sixteenths_l3755_375551

/-- Represents the dartboard with inner and outer circles and point values -/
structure Dartboard where
  inner_radius : ℝ
  outer_radius : ℝ
  inner_values : Fin 3 → ℕ
  outer_values : Fin 3 → ℕ

/-- Calculates the probability of getting an even sum with two darts -/
def prob_even_sum (d : Dartboard) : ℚ :=
  sorry

/-- The specific dartboard described in the problem -/
def problem_dartboard : Dartboard where
  inner_radius := 4
  outer_radius := 8
  inner_values := ![3, 5, 5]
  outer_values := ![4, 3, 3]

theorem prob_even_sum_is_seven_sixteenths :
  prob_even_sum problem_dartboard = 7/16 := by
  sorry

end prob_even_sum_is_seven_sixteenths_l3755_375551


namespace log_equality_implies_y_value_l3755_375556

-- Define the logarithm relationship
def log_relation (m y : ℝ) : Prop :=
  (Real.log y / Real.log m) * (Real.log m / Real.log 7) = 4

-- Theorem statement
theorem log_equality_implies_y_value :
  ∀ m y : ℝ, m > 0 ∧ m ≠ 1 ∧ y > 0 → log_relation m y → y = 2401 :=
by
  sorry

end log_equality_implies_y_value_l3755_375556


namespace smallest_two_digit_multiple_l3755_375594

theorem smallest_two_digit_multiple : ∃ n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ k : ℕ, n = 30 * k + 2) ∧
  (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ (∃ j : ℕ, m = 30 * j + 2) → m ≥ n) ∧
  n = 32 := by
sorry

end smallest_two_digit_multiple_l3755_375594


namespace smallest_number_l3755_375519

theorem smallest_number (a b c d : ℚ) (ha : a = 1) (hb : b = -2) (hc : c = 0) (hd : d = -5/2) :
  d < b ∧ b < c ∧ c < a := by
  sorry

end smallest_number_l3755_375519


namespace number_puzzle_solution_l3755_375546

theorem number_puzzle_solution (A B C : ℤ) 
  (sum_eq : A + B = 44)
  (ratio_eq : 5 * A = 6 * B)
  (diff_eq : C = 2 * (A - B)) :
  A = 24 ∧ B = 20 ∧ C = 8 := by
sorry

end number_puzzle_solution_l3755_375546


namespace least_three_digit_multiple_l3755_375505

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 105 :=
by sorry

end least_three_digit_multiple_l3755_375505


namespace amendment_effects_l3755_375552

-- Define the administrative actions included in the amendment
def administrative_actions : Set String := 
  {"abuse of administrative power", "illegal fundraising", "apportionment of expenses", "failure to pay benefits"}

-- Define the amendment to the Administrative Litigation Law
def administrative_litigation_amendment (actions : Set String) : Prop :=
  ∀ action ∈ actions, action ∈ administrative_actions

-- Define the concept of standardizing government power exercise
def standardizes_government_power (amendment : Set String → Prop) : Prop :=
  amendment administrative_actions → 
    ∃ standard : String, standard = "improved government power exercise"

-- Define the concept of protecting citizens' rights
def protects_citizens_rights (amendment : Set String → Prop) : Prop :=
  amendment administrative_actions → 
    ∃ protection : String, protection = "better protection of citizens' rights"

-- Theorem statement
theorem amendment_effects 
  (h : administrative_litigation_amendment administrative_actions) :
  standardizes_government_power administrative_litigation_amendment ∧ 
  protects_citizens_rights administrative_litigation_amendment :=
by sorry

end amendment_effects_l3755_375552


namespace vector_subtraction_magnitude_l3755_375588

theorem vector_subtraction_magnitude : ∃ (a b : ℝ × ℝ), 
  a = (2, 1) ∧ b = (-2, 4) ∧ 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 := by
  sorry

end vector_subtraction_magnitude_l3755_375588


namespace robert_reading_capacity_l3755_375595

/-- Represents the number of books Robert can read in a given time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (pages_per_hour * available_hours) / pages_per_book

/-- Theorem stating that Robert can read 2 books in 6 hours -/
theorem robert_reading_capacity : books_read 90 270 6 = 2 := by
  sorry

end robert_reading_capacity_l3755_375595


namespace percentage_of_english_books_l3755_375530

theorem percentage_of_english_books (total_books : ℕ) 
  (english_books_outside : ℕ) (percentage_published_in_country : ℚ) :
  total_books = 2300 →
  english_books_outside = 736 →
  percentage_published_in_country = 60 / 100 →
  (english_books_outside / (1 - percentage_published_in_country)) / total_books = 80 / 100 :=
by
  sorry

end percentage_of_english_books_l3755_375530


namespace complex_in_second_quadrant_l3755_375567

theorem complex_in_second_quadrant (m : ℝ) :
  let z : ℂ := (2 + m * I) / (4 - 5 * I)
  (z.re < 0 ∧ z.im > 0) ↔ m > 8/5 := by sorry

end complex_in_second_quadrant_l3755_375567


namespace distance_to_line_l3755_375550

/-- Represents a square with side length 2 inches -/
structure Square where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- The configuration of three squares, where the middle one is rotated -/
structure SquareConfiguration where
  left : Square
  middle : Square
  right : Square
  middle_rotated : middle.side_length = left.side_length ∧ middle.side_length = right.side_length

/-- The theorem stating the distance of point B from the original line -/
theorem distance_to_line (config : SquareConfiguration) :
  let diagonal := config.middle.side_length * Real.sqrt 2
  let height_increase := diagonal / 2
  let original_height := config.middle.side_length / 2
  height_increase + original_height = 2 * Real.sqrt 2 := by sorry

end distance_to_line_l3755_375550


namespace correct_sampling_pairing_l3755_375583

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the sampling scenarios
structure SamplingScenario where
  description : String
  populationSize : Nat
  sampleSize : Nat
  hasStrata : Bool

-- Define the correct pairing function
def correctPairing (scenario : SamplingScenario) : SamplingMethod :=
  if scenario.hasStrata then
    SamplingMethod.Stratified
  else if scenario.populationSize ≤ 100 then
    SamplingMethod.SimpleRandom
  else
    SamplingMethod.Systematic

-- Define the three scenarios
def universityScenario : SamplingScenario :=
  { description := "University student sampling"
  , populationSize := 300
  , sampleSize := 100
  , hasStrata := true }

def productScenario : SamplingScenario :=
  { description := "Product quality inspection"
  , populationSize := 20
  , sampleSize := 7
  , hasStrata := false }

def habitScenario : SamplingScenario :=
  { description := "Daily habits sampling"
  , populationSize := 2000
  , sampleSize := 10
  , hasStrata := false }

-- Theorem statement
theorem correct_sampling_pairing :
  (correctPairing universityScenario = SamplingMethod.Stratified) ∧
  (correctPairing productScenario = SamplingMethod.SimpleRandom) ∧
  (correctPairing habitScenario = SamplingMethod.Systematic) :=
sorry

end correct_sampling_pairing_l3755_375583


namespace square_difference_l3755_375554

theorem square_difference (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : a^2 - b^2 = 15 := by
  sorry

end square_difference_l3755_375554


namespace farm_animals_l3755_375571

theorem farm_animals (goats chickens ducks pigs : ℕ) : 
  chickens = 2 * goats →
  ducks = (goats + chickens) / 2 →
  pigs = ducks / 3 →
  goats = pigs + 33 →
  goats = 66 := by
sorry

end farm_animals_l3755_375571


namespace michaels_initial_money_l3755_375592

/-- 
Given:
- Michael's brother initially had $17
- Michael gave half of his money to his brother
- His brother spent $3 on candy
- His brother had $35 left after buying candy

Prove that Michael's initial amount of money was $42
-/
theorem michaels_initial_money (brother_initial : ℕ) (candy_cost : ℕ) (brother_final : ℕ) :
  brother_initial = 17 →
  candy_cost = 3 →
  brother_final = 35 →
  ∃ (michael_initial : ℕ), 
    brother_initial + michael_initial / 2 = brother_final + candy_cost ∧
    michael_initial = 42 := by
  sorry

end michaels_initial_money_l3755_375592


namespace apple_purchase_difference_l3755_375572

theorem apple_purchase_difference : 
  ∀ (bonnie_apples samuel_apples : ℕ),
    bonnie_apples = 8 →
    samuel_apples > bonnie_apples →
    samuel_apples - (samuel_apples / 2) - (samuel_apples / 7) = 10 →
    samuel_apples - bonnie_apples = 20 := by
  sorry

end apple_purchase_difference_l3755_375572


namespace ice_cream_cost_l3755_375504

theorem ice_cream_cost (two_cones_cost : ℕ) (h : two_cones_cost = 198) : 
  two_cones_cost / 2 = 99 := by sorry

end ice_cream_cost_l3755_375504


namespace norris_remaining_money_l3755_375585

/-- Calculates the remaining money for Norris after savings and spending --/
theorem norris_remaining_money 
  (september_savings : ℕ) 
  (october_savings : ℕ) 
  (november_savings : ℕ) 
  (game_cost : ℕ) : 
  september_savings = 29 →
  october_savings = 25 →
  november_savings = 31 →
  game_cost = 75 →
  (september_savings + october_savings + november_savings) - game_cost = 10 := by
  sorry

end norris_remaining_money_l3755_375585


namespace linear_function_decreasing_l3755_375566

theorem linear_function_decreasing (a b y₁ y₂ : ℝ) :
  a < 0 →
  y₁ = 2 * a * (-1) - b →
  y₂ = 2 * a * 2 - b →
  y₁ > y₂ := by
  sorry

end linear_function_decreasing_l3755_375566


namespace sin_product_identity_l3755_375598

theorem sin_product_identity : 
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 1/8 := by
  sorry

end sin_product_identity_l3755_375598
