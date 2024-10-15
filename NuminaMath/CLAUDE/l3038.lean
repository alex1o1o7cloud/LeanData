import Mathlib

namespace NUMINAMATH_CALUDE_maryGarbageBillIs102_l3038_303873

/-- Calculates Mary's monthly garbage bill --/
def maryGarbageBill : ℚ :=
  let trashBinCharge : ℚ := 10
  let recyclingBinCharge : ℚ := 5
  let trashBinCount : ℕ := 2
  let recyclingBinCount : ℕ := 1
  let weeksInMonth : ℕ := 4
  let elderlyDiscountPercentage : ℚ := 18 / 100
  let inappropriateItemsFine : ℚ := 20

  let weeklyCharge := trashBinCharge * trashBinCount + recyclingBinCharge * recyclingBinCount
  let monthlyCharge := weeklyCharge * weeksInMonth
  let discountAmount := monthlyCharge * elderlyDiscountPercentage
  let discountedMonthlyCharge := monthlyCharge - discountAmount
  discountedMonthlyCharge + inappropriateItemsFine

theorem maryGarbageBillIs102 : maryGarbageBill = 102 := by
  sorry

end NUMINAMATH_CALUDE_maryGarbageBillIs102_l3038_303873


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3038_303846

theorem quadratic_inequality_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3038_303846


namespace NUMINAMATH_CALUDE_complement_A_union_B_subset_l3038_303879

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_A_union_B_subset :
  (Set.compl A ∪ B) ⊆ {x : ℝ | x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_subset_l3038_303879


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3038_303881

def is_solution (x y z : ℕ+) : Prop :=
  (x + y) * (y + z) * (z + x) = x * y * z * (x + y + z) ∧
  Nat.gcd x.val y.val = 1 ∧ Nat.gcd y.val z.val = 1 ∧ Nat.gcd z.val x.val = 1

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+, is_solution x y z ↔ 
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ 
     (x = 1 ∧ y = 1 ∧ z = 2) ∨ 
     (x = 1 ∧ y = 2 ∧ z = 3)) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3038_303881


namespace NUMINAMATH_CALUDE_correct_balloons_left_l3038_303877

/-- Given the number of balloons of each color and the number of friends,
    calculate the number of balloons left after even distribution. -/
def balloons_left (yellow blue pink violet friends : ℕ) : ℕ :=
  let total := yellow + blue + pink + violet
  total % friends

theorem correct_balloons_left :
  balloons_left 20 24 50 102 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_balloons_left_l3038_303877


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l3038_303850

/-- The sum of an infinite geometric series with first term 5/3 and common ratio 1/3 is 5/2. -/
theorem infinite_geometric_series_sum :
  let a : ℚ := 5/3  -- First term
  let r : ℚ := 1/3  -- Common ratio
  let S : ℚ := a / (1 - r)  -- Sum formula for infinite geometric series
  S = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l3038_303850


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3038_303885

def polynomial (x : ℝ) : ℝ := 4 * (2 * x^6 + 9 * x^3 - 6) + 8 * (x^4 - 6 * x^2 + 3)

theorem sum_of_coefficients : (polynomial 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3038_303885


namespace NUMINAMATH_CALUDE_anna_lettuce_plants_l3038_303831

/-- The number of large salads Anna wants --/
def desired_salads : ℕ := 12

/-- The fraction of lettuce that will be lost --/
def loss_fraction : ℚ := 1/2

/-- The number of large salads each lettuce plant provides --/
def salads_per_plant : ℕ := 3

/-- The number of lettuce plants Anna should grow --/
def plants_to_grow : ℕ := 8

theorem anna_lettuce_plants : 
  (plants_to_grow : ℚ) * (1 - loss_fraction) * salads_per_plant ≥ desired_salads := by
  sorry

end NUMINAMATH_CALUDE_anna_lettuce_plants_l3038_303831


namespace NUMINAMATH_CALUDE_normalized_coordinates_sum_of_squares_is_one_l3038_303822

/-- The sum of squares of normalized coordinates is 1 -/
theorem normalized_coordinates_sum_of_squares_is_one
  (a b : ℝ) -- Coordinates of point Q
  (d : ℝ) -- Distance from origin to Q
  (h_d : d = Real.sqrt (a^2 + b^2)) -- Definition of distance
  (u : ℝ) (h_u : u = b / d) -- Definition of u
  (v : ℝ) (h_v : v = a / d) -- Definition of v
  : u^2 + v^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_normalized_coordinates_sum_of_squares_is_one_l3038_303822


namespace NUMINAMATH_CALUDE_irene_age_is_46_l3038_303852

/-- Given Eddie's age, calculate Irene's age based on the relationships between Eddie, Becky, and Irene. -/
def calculate_irene_age (eddie_age : ℕ) : ℕ :=
  let becky_age := eddie_age / 4
  2 * becky_age

/-- Theorem stating that given the conditions, Irene's age is 46. -/
theorem irene_age_is_46 :
  let eddie_age : ℕ := 92
  calculate_irene_age eddie_age = 46 := by
  sorry

#eval calculate_irene_age 92

end NUMINAMATH_CALUDE_irene_age_is_46_l3038_303852


namespace NUMINAMATH_CALUDE_total_owls_on_fence_l3038_303854

def initial_owls : ℕ := 3
def joining_owls : ℕ := 2

theorem total_owls_on_fence : initial_owls + joining_owls = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_owls_on_fence_l3038_303854


namespace NUMINAMATH_CALUDE_average_book_cost_l3038_303802

def initial_amount : ℕ := 236
def books_bought : ℕ := 6
def remaining_amount : ℕ := 14

theorem average_book_cost :
  (initial_amount - remaining_amount) / books_bought = 37 := by
  sorry

end NUMINAMATH_CALUDE_average_book_cost_l3038_303802


namespace NUMINAMATH_CALUDE_am_gm_inequality_l3038_303839

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hab : a < b) (hbc : b < c) :
  ((a + c) / 2) - Real.sqrt (a * c) < (c - a)^2 / (8 * a) := by
sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l3038_303839


namespace NUMINAMATH_CALUDE_least_stamps_stamps_23_robert_stamps_l3038_303865

theorem least_stamps (n : ℕ) : n > 0 ∧ n % 7 = 2 ∧ n % 4 = 3 → n ≥ 23 := by
  sorry

theorem stamps_23 : 23 % 7 = 2 ∧ 23 % 4 = 3 := by
  sorry

theorem robert_stamps : ∃ n : ℕ, n > 0 ∧ n % 7 = 2 ∧ n % 4 = 3 ∧ 
  ∀ m : ℕ, (m > 0 ∧ m % 7 = 2 ∧ m % 4 = 3) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_stamps_stamps_23_robert_stamps_l3038_303865


namespace NUMINAMATH_CALUDE_f_of_5_equals_0_l3038_303829

theorem f_of_5_equals_0 (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = x^2 - 2*x) : f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_0_l3038_303829


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3038_303812

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3038_303812


namespace NUMINAMATH_CALUDE_yearly_reading_pages_l3038_303838

/-- The number of pages read in a year, given the number of novels read per month,
    pages per novel, and months in a year. -/
def pages_read_in_year (novels_per_month : ℕ) (pages_per_novel : ℕ) (months_in_year : ℕ) : ℕ :=
  novels_per_month * pages_per_novel * months_in_year

/-- Theorem stating that reading 4 novels of 200 pages each month for 12 months
    results in reading 9600 pages in a year. -/
theorem yearly_reading_pages :
  pages_read_in_year 4 200 12 = 9600 := by
  sorry

end NUMINAMATH_CALUDE_yearly_reading_pages_l3038_303838


namespace NUMINAMATH_CALUDE_magician_min_earnings_l3038_303813

/-- Represents the earnings of a magician selling card decks --/
def magician_earnings (initial_decks : ℕ) (remaining_decks : ℕ) (full_price : ℕ) (discounted_price : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * discounted_price

/-- Theorem stating the minimum earnings of the magician --/
theorem magician_min_earnings :
  let initial_decks : ℕ := 15
  let remaining_decks : ℕ := 3
  let full_price : ℕ := 3
  let discounted_price : ℕ := 2
  magician_earnings initial_decks remaining_decks full_price discounted_price ≥ 24 := by
  sorry

#check magician_min_earnings

end NUMINAMATH_CALUDE_magician_min_earnings_l3038_303813


namespace NUMINAMATH_CALUDE_expected_voters_for_candidate_A_prove_expected_voters_for_candidate_A_l3038_303845

theorem expected_voters_for_candidate_A : ℝ → Prop :=
  fun x => 
    -- Define the percentage of Democrats
    let percent_democrats : ℝ := 0.60
    -- Define the percentage of Republicans
    let percent_republicans : ℝ := 1 - percent_democrats
    -- Define the percentage of Democrats voting for A
    let percent_democrats_for_A : ℝ := 0.85
    -- Define the percentage of Republicans voting for A
    let percent_republicans_for_A : ℝ := 0.20
    -- Calculate the total percentage of voters for A
    let total_percent_for_A : ℝ := 
      percent_democrats * percent_democrats_for_A + 
      percent_republicans * percent_republicans_for_A
    -- The theorem statement
    x = total_percent_for_A * 100

-- The proof of the theorem
theorem prove_expected_voters_for_candidate_A : 
  expected_voters_for_candidate_A 59 := by
  sorry

end NUMINAMATH_CALUDE_expected_voters_for_candidate_A_prove_expected_voters_for_candidate_A_l3038_303845


namespace NUMINAMATH_CALUDE_friends_to_movies_l3038_303876

theorem friends_to_movies (total_friends : ℕ) (cant_go : ℕ) (can_go : ℕ) 
  (h1 : total_friends = 15)
  (h2 : cant_go = 7)
  (h3 : can_go = total_friends - cant_go) :
  can_go = 8 := by
  sorry

end NUMINAMATH_CALUDE_friends_to_movies_l3038_303876


namespace NUMINAMATH_CALUDE_cos_150_degrees_l3038_303889

theorem cos_150_degrees : Real.cos (150 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l3038_303889


namespace NUMINAMATH_CALUDE_unique_solution_power_sum_l3038_303803

theorem unique_solution_power_sum (a b c : ℕ) :
  (∀ n : ℕ, a^n + b^n = c^(n+1)) → (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_power_sum_l3038_303803


namespace NUMINAMATH_CALUDE_bite_size_samples_per_half_l3038_303882

def total_pies : ℕ := 13
def halves_per_pie : ℕ := 2
def total_tasters : ℕ := 130

theorem bite_size_samples_per_half : 
  (total_tasters / (total_pies * halves_per_pie) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_bite_size_samples_per_half_l3038_303882


namespace NUMINAMATH_CALUDE_hexagon_ratio_theorem_l3038_303860

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  /-- Total area of the hexagon in square units -/
  total_area : ℝ
  /-- Width of the hexagon -/
  width : ℝ
  /-- Height of the rectangle below PQ -/
  rect_height : ℝ
  /-- Area below PQ -/
  area_below_pq : ℝ
  /-- Ensures the hexagon consists of 7 unit squares -/
  area_constraint : total_area = 7
  /-- Ensures PQ bisects the hexagon area -/
  bisect_constraint : area_below_pq = total_area / 2
  /-- Ensures the triangle base is half the hexagon width -/
  triangle_base_constraint : width / 2 = width - (width / 2)

/-- The main theorem to prove -/
theorem hexagon_ratio_theorem (h : Hexagon) : 
  let xq := (h.area_below_pq - h.width * h.rect_height) / (h.width / 4)
  let qy := h.width - xq
  xq / qy = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_ratio_theorem_l3038_303860


namespace NUMINAMATH_CALUDE_equal_probabilities_after_adding_balls_l3038_303899

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  white : ℕ
  yellow : ℕ

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (bag : BagContents) (color : ℕ) : ℚ :=
  color / (bag.white + bag.yellow)

/-- The initial contents of the bag -/
def initialBag : BagContents := ⟨2, 3⟩

/-- The contents of the bag after adding balls -/
def finalBag : BagContents := ⟨initialBag.white + 4, initialBag.yellow + 3⟩

/-- Theorem stating that the probabilities are equal after adding balls -/
theorem equal_probabilities_after_adding_balls :
  probability finalBag finalBag.white = probability finalBag finalBag.yellow := by
  sorry

end NUMINAMATH_CALUDE_equal_probabilities_after_adding_balls_l3038_303899


namespace NUMINAMATH_CALUDE_ivan_walking_time_l3038_303835

/-- Represents the problem of determining how long Ivan Ivanovich walked. -/
theorem ivan_walking_time 
  (s : ℝ) -- Total distance from home to work
  (t : ℝ) -- Usual time taken by car
  (v : ℝ) -- Car's speed
  (u : ℝ) -- Ivan's walking speed
  (h1 : s = v * t) -- Total distance equals car speed times usual time
  (h2 : s = u * T + v * (t - T + 1/6)) -- Distance covered by walking and car
  (h3 : v * (1/12) = s - u * T) -- Car meets Ivan halfway through its usual journey
  (h4 : v > 0) -- Car speed is positive
  (h5 : u > 0) -- Walking speed is positive
  (h6 : v > u) -- Car is faster than walking
  : T = 55 := by
  sorry

#check ivan_walking_time

end NUMINAMATH_CALUDE_ivan_walking_time_l3038_303835


namespace NUMINAMATH_CALUDE_max_consecutive_sum_l3038_303880

theorem max_consecutive_sum (n : ℕ) : (n * (n + 1)) / 2 ≤ 1000 ↔ n ≤ 44 := by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_l3038_303880


namespace NUMINAMATH_CALUDE_point_transformation_l3038_303841

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = -x -/
def reflectAboutNegX (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem point_transformation (c d : ℝ) : 
  let (x₁, y₁) := rotate90 c d 2 3
  let (x₂, y₂) := reflectAboutNegX x₁ y₁
  (x₂ = 7 ∧ y₂ = -10) → d - c = -7 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3038_303841


namespace NUMINAMATH_CALUDE_x_times_x_minus_one_eq_six_is_quadratic_l3038_303808

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x(x-1) = 6 -/
def f (x : ℝ) : ℝ := x * (x - 1) - 6

theorem x_times_x_minus_one_eq_six_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_times_x_minus_one_eq_six_is_quadratic_l3038_303808


namespace NUMINAMATH_CALUDE_remainder_problem_l3038_303858

theorem remainder_problem (x : ℕ) :
  x < 100 →
  x % 3 = 2 →
  x % 4 = 2 →
  x % 5 = 2 →
  x = 2 ∨ x = 62 :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3038_303858


namespace NUMINAMATH_CALUDE_quadratic_functions_problem_l3038_303866

/-- Given two quadratic functions y₁ and y₂ satisfying certain conditions, 
    prove that α = 1, y₁ = -2x² + 4x + 3, and y₂ = 3x² + 12x + 10 -/
theorem quadratic_functions_problem 
  (y₁ y₂ : ℝ → ℝ) 
  (α : ℝ) 
  (h_α_pos : α > 0)
  (h_y₁_max : ∀ x, y₁ x ≤ y₁ α)
  (h_y₁_max_val : y₁ α = 5)
  (h_y₂_α : y₂ α = 25)
  (h_y₂_min : ∀ x, y₂ x ≥ -2)
  (h_sum : ∀ x, y₁ x + y₂ x = x^2 + 16*x + 13) :
  α = 1 ∧ 
  (∀ x, y₁ x = -2*x^2 + 4*x + 3) ∧ 
  (∀ x, y₂ x = 3*x^2 + 12*x + 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_functions_problem_l3038_303866


namespace NUMINAMATH_CALUDE_hyperbola_center_is_two_two_l3038_303857

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * y - 8)^2 / 8^2 - (5 * x - 10)^2 / 7^2 = 1

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (2, 2)

/-- Theorem: The center of the given hyperbola is (2, 2) -/
theorem hyperbola_center_is_two_two :
  ∀ x y : ℝ, hyperbola_equation x y → hyperbola_center = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_two_two_l3038_303857


namespace NUMINAMATH_CALUDE_cube_with_cut_corners_has_36_edges_l3038_303883

/-- A cube with cut corners is a polyhedron resulting from cutting off each corner of a cube
    such that the cutting planes do not intersect within or on the cube. -/
structure CubeWithCutCorners where
  -- We don't need to define the structure explicitly for this problem

/-- The number of edges in a cube with cut corners -/
def num_edges_cube_with_cut_corners : ℕ := 36

/-- Theorem stating that a cube with cut corners has 36 edges -/
theorem cube_with_cut_corners_has_36_edges (c : CubeWithCutCorners) :
  num_edges_cube_with_cut_corners = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_cut_corners_has_36_edges_l3038_303883


namespace NUMINAMATH_CALUDE_number_of_one_point_two_stamps_l3038_303834

/-- Represents the number of stamps of each denomination -/
structure StampCounts where
  half : ℕ
  eightyPercent : ℕ
  onePointTwo : ℕ

/-- The total value of all stamps in cents -/
def totalValue (s : StampCounts) : ℕ :=
  50 * s.half + 80 * s.eightyPercent + 120 * s.onePointTwo

/-- The theorem stating the number of 1.2 yuan stamps given the conditions -/
theorem number_of_one_point_two_stamps :
  ∃ (s : StampCounts),
    totalValue s = 6000 ∧
    s.eightyPercent = 4 * s.half ∧
    s.onePointTwo = 13 :=
by sorry

end NUMINAMATH_CALUDE_number_of_one_point_two_stamps_l3038_303834


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3038_303867

/-- Given a hyperbola with the following properties:
    1. A line is drawn through the left focus F₁ at a 30° angle
    2. This line intersects the right branch of the hyperbola at point P
    3. A circle with diameter PF₁ passes through the right focus F₂
    Then the eccentricity of the hyperbola is √3 -/
theorem hyperbola_eccentricity (F₁ F₂ P : ℝ × ℝ) (a b c : ℝ) :
  let e := c / a
  (P.1 = c ∧ P.2 = b^2 / a) →  -- P is on the right branch
  (P.2 / (2 * c) = Real.tan (30 * π / 180)) →  -- Line through F₁ is at 30°
  (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)) →  -- Circle condition
  e = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3038_303867


namespace NUMINAMATH_CALUDE_pears_for_apples_l3038_303826

/-- The cost of fruits in a common unit -/
structure FruitCost where
  apple : ℕ
  orange : ℕ
  pear : ℕ

/-- The relationship between apple and orange costs -/
def apple_orange_relation (fc : FruitCost) : Prop :=
  10 * fc.apple = 5 * fc.orange

/-- The relationship between orange and pear costs -/
def orange_pear_relation (fc : FruitCost) : Prop :=
  4 * fc.orange = 6 * fc.pear

/-- The main theorem: Nancy can buy 15 pears for the price of 20 apples -/
theorem pears_for_apples (fc : FruitCost) 
  (h1 : apple_orange_relation fc) 
  (h2 : orange_pear_relation fc) : 
  20 * fc.apple = 15 * fc.pear :=
by sorry

end NUMINAMATH_CALUDE_pears_for_apples_l3038_303826


namespace NUMINAMATH_CALUDE_sum_first_eight_super_nice_l3038_303870

def is_prime (n : ℕ) : Prop := sorry

def is_super_nice (n : ℕ) : Prop :=
  (∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r) ∨
  (∃ p : ℕ, is_prime p ∧ n = p^4)

def first_eight_super_nice : List ℕ :=
  [16, 30, 42, 66, 70, 81, 105, 110]

theorem sum_first_eight_super_nice :
  (∀ n ∈ first_eight_super_nice, is_super_nice n) ∧
  (∀ m : ℕ, m < 16 → ¬is_super_nice m) ∧
  (∀ m : ℕ, m > 110 ∧ is_super_nice m → ∃ n ∈ first_eight_super_nice, m > n) ∧
  (List.sum first_eight_super_nice = 520) :=
by sorry

end NUMINAMATH_CALUDE_sum_first_eight_super_nice_l3038_303870


namespace NUMINAMATH_CALUDE_max_value_when_a_zero_one_zero_iff_a_positive_l3038_303809

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Part 1: Maximum value when a = 0
theorem max_value_when_a_zero :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
sorry

-- Part 2: Range of a for exactly one zero
theorem one_zero_iff_a_positive :
  ∀ (a : ℝ), (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end

end NUMINAMATH_CALUDE_max_value_when_a_zero_one_zero_iff_a_positive_l3038_303809


namespace NUMINAMATH_CALUDE_remainder_theorem_l3038_303872

theorem remainder_theorem (P D Q R Q' R' : ℕ) (hD : D > 1) 
  (h1 : P = Q * D + R) (h2 : Q = (D - 1) * Q' + R') :
  P % (D * (D - 1)) = D * R' + R :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3038_303872


namespace NUMINAMATH_CALUDE_berry_pie_theorem_l3038_303836

/-- Represents the amount of berries picked by each person -/
structure BerryPicker where
  strawberries : ℕ
  blueberries : ℕ
  raspberries : ℕ

/-- Represents the requirements for each type of pie -/
structure PieRequirements where
  strawberry : ℕ
  blueberry : ℕ
  raspberry : ℕ

/-- Calculates the maximum number of complete pies that can be made -/
def max_pies (christine : BerryPicker) (rachel : BerryPicker) (req : PieRequirements) : ℕ × ℕ × ℕ :=
  let total_strawberries := christine.strawberries + rachel.strawberries
  let total_blueberries := christine.blueberries + rachel.blueberries
  let total_raspberries := christine.raspberries + rachel.raspberries
  (total_strawberries / req.strawberry,
   total_blueberries / req.blueberry,
   total_raspberries / req.raspberry)

theorem berry_pie_theorem (christine : BerryPicker) (rachel : BerryPicker) (req : PieRequirements) :
  christine.strawberries = 10 ∧
  christine.blueberries = 8 ∧
  christine.raspberries = 20 ∧
  rachel.strawberries = 2 * christine.strawberries ∧
  rachel.blueberries = 2 * christine.blueberries ∧
  rachel.raspberries = christine.raspberries / 2 ∧
  req.strawberry = 3 ∧
  req.blueberry = 2 ∧
  req.raspberry = 4 →
  max_pies christine rachel req = (10, 12, 7) := by
  sorry

end NUMINAMATH_CALUDE_berry_pie_theorem_l3038_303836


namespace NUMINAMATH_CALUDE_quadratic_solutions_l3038_303856

theorem quadratic_solutions : 
  (∃ (x : ℝ), x^2 - 8*x + 12 = 0) ∧ 
  (∃ (x : ℝ), x^2 - 2*x - 8 = 0) ∧ 
  ({x : ℝ | x^2 - 8*x + 12 = 0} = {2, 6}) ∧
  ({x : ℝ | x^2 - 2*x - 8 = 0} = {-2, 4}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solutions_l3038_303856


namespace NUMINAMATH_CALUDE_partner_a_share_l3038_303859

/-- Calculates the share of a partner in a partnership based on investments and known share of another partner. -/
def calculate_share (investment_a investment_b investment_c : ℚ) (share_b : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  let ratio_b := investment_b / total_investment
  let total_profit := share_b / ratio_b
  ratio_a * total_profit

/-- Theorem stating that given the investments and b's share, a's share is approximately $560. -/
theorem partner_a_share (investment_a investment_b investment_c share_b : ℚ) 
  (h1 : investment_a = 7000)
  (h2 : investment_b = 11000)
  (h3 : investment_c = 18000)
  (h4 : share_b = 880) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_share investment_a investment_b investment_c share_b - 560| < ε :=
sorry

end NUMINAMATH_CALUDE_partner_a_share_l3038_303859


namespace NUMINAMATH_CALUDE_expand_expression_l3038_303897

theorem expand_expression (x : ℝ) : (2 * x + 1) * (x - 2) = 2 * x^2 - 3 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3038_303897


namespace NUMINAMATH_CALUDE_exam_pass_count_l3038_303801

theorem exam_pass_count :
  let total_candidates : ℕ := 120
  let overall_average : ℚ := 35
  let pass_average : ℚ := 39
  let fail_average : ℚ := 15
  ∃ pass_count : ℕ,
    pass_count ≤ total_candidates ∧
    (pass_count : ℚ) * pass_average + (total_candidates - pass_count : ℚ) * fail_average = 
      (total_candidates : ℚ) * overall_average ∧
    pass_count = 100 := by
sorry

end NUMINAMATH_CALUDE_exam_pass_count_l3038_303801


namespace NUMINAMATH_CALUDE_fourth_month_sales_l3038_303847

def sales_month1 : ℕ := 3435
def sales_month2 : ℕ := 3920
def sales_month3 : ℕ := 3855
def sales_month5 : ℕ := 3560
def sales_month6 : ℕ := 2000
def average_sale : ℕ := 3500
def num_months : ℕ := 6

theorem fourth_month_sales :
  ∃ (sales_month4 : ℕ),
    sales_month4 = 4230 ∧
    (sales_month1 + sales_month2 + sales_month3 + sales_month4 + sales_month5 + sales_month6) / num_months = average_sale :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l3038_303847


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l3038_303848

def euler_family_ages : List ℕ := [6, 6, 6, 6, 8, 8, 16]

theorem euler_family_mean_age :
  (euler_family_ages.sum / euler_family_ages.length : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l3038_303848


namespace NUMINAMATH_CALUDE_count_valid_integers_valid_integers_formula_correct_l3038_303820

/-- The number of n-digit decimal integers using only digits 1, 2, and 3,
    and containing each of these digits at least once. -/
def validIntegers (n : ℕ+) : ℕ :=
  3^n.val - 3 * 2^n.val + 3

/-- Theorem stating that validIntegers gives the correct count. -/
theorem count_valid_integers (n : ℕ+) :
  validIntegers n = (3^n.val - 3 * 2^n.val + 3) := by
  sorry

/-- Proof that the formula is correct for all positive integers n. -/
theorem valid_integers_formula_correct :
  ∀ n : ℕ+, validIntegers n = (3^n.val - 3 * 2^n.val + 3) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_integers_valid_integers_formula_correct_l3038_303820


namespace NUMINAMATH_CALUDE_book_sharing_probability_l3038_303824

/-- The number of students sharing books -/
def num_students : ℕ := 2

/-- The number of books being shared -/
def num_books : ℕ := 3

/-- The total number of possible book distribution scenarios -/
def total_scenarios : ℕ := 8

/-- The number of scenarios where one student gets all books and the other gets none -/
def favorable_scenarios : ℕ := 2

/-- The probability of one student getting all books and the other getting none -/
def probability : ℚ := favorable_scenarios / total_scenarios

theorem book_sharing_probability :
  probability = 1/4 := by sorry

end NUMINAMATH_CALUDE_book_sharing_probability_l3038_303824


namespace NUMINAMATH_CALUDE_davids_original_portion_l3038_303868

/-- Given a total initial amount of $1500 shared among David, Elisa, and Frank,
    where the total final amount is $2700, Elisa and Frank both triple their initial investments,
    and David loses $200, prove that David's original portion is $800. -/
theorem davids_original_portion (d e f : ℝ) : 
  d + e + f = 1500 →
  d - 200 + 3 * e + 3 * f = 2700 →
  d = 800 := by
sorry

end NUMINAMATH_CALUDE_davids_original_portion_l3038_303868


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l3038_303896

/-- The maximum exponent for the line segment lengths -/
def max_exponent : ℕ := 10

/-- The set of line segment lengths -/
def segment_lengths : Set ℕ := {n | ∃ k : ℕ, 0 ≤ k ∧ k ≤ max_exponent ∧ n = 2^k}

/-- A function to check if three lengths can form a nondegenerate triangle -/
def is_nondegenerate_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The number of distinct nondegenerate triangles -/
def num_distinct_triangles : ℕ := Nat.choose (max_exponent + 1) 2

theorem distinct_triangles_count :
  num_distinct_triangles = 55 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_count_l3038_303896


namespace NUMINAMATH_CALUDE_price_change_calculation_l3038_303874

theorem price_change_calculation :
  let original_price := 100
  let price_after_day1 := original_price * (1 - 0.12)
  let price_after_day2 := price_after_day1 * (1 - 0.10)
  let price_after_day3 := price_after_day2 * (1 - 0.08)
  let final_price := price_after_day3 * (1 + 0.05)
  (final_price / original_price) * 100 = 76.5072 := by
sorry

end NUMINAMATH_CALUDE_price_change_calculation_l3038_303874


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3038_303842

theorem binomial_coefficient_congruence 
  (p : Nat) 
  (hp : p.Prime ∧ p > 3 ∧ Odd p) 
  (a b : Nat) 
  (hab : a > b ∧ b > 1) : 
  Nat.choose (a * p) (a * p) ≡ Nat.choose a b [MOD p^3] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3038_303842


namespace NUMINAMATH_CALUDE_bar_charts_cannot_show_change_line_charts_can_show_change_bar_charts_show_change_is_false_l3038_303844

-- Define the types of charts
inductive Chart
| Bar
| Line

-- Define the capability of showing increase or decrease
def can_show_change (c : Chart) : Prop :=
  match c with
  | Chart.Line => true
  | Chart.Bar => false

-- Theorem stating that bar charts cannot show change
theorem bar_charts_cannot_show_change :
  ¬(can_show_change Chart.Bar) :=
by
  sorry

-- Theorem stating that line charts can show change
theorem line_charts_can_show_change :
  can_show_change Chart.Line :=
by
  sorry

-- Main theorem proving the original statement is false
theorem bar_charts_show_change_is_false :
  ¬(∀ (c : Chart), can_show_change c) :=
by
  sorry

end NUMINAMATH_CALUDE_bar_charts_cannot_show_change_line_charts_can_show_change_bar_charts_show_change_is_false_l3038_303844


namespace NUMINAMATH_CALUDE_parabola_through_fixed_point_l3038_303861

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := (a - 1) * x - y + 2 * a + 1 = 0

-- Define the fixed point P
def fixed_point : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem parabola_through_fixed_point :
  (∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2)) ∧
  (∃ p : ℝ, p > 0 ∧ 
    ((∀ x y : ℝ, (x, y) = fixed_point → y^2 = -2*p*x) ∨
     (∀ x y : ℝ, (x, y) = fixed_point → x^2 = 2*p*y))) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_fixed_point_l3038_303861


namespace NUMINAMATH_CALUDE_smallest_factorial_divisible_by_23m_and_33n_l3038_303821

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_factorial_divisible_by_23m_and_33n :
  (∀ k < 24, ¬(factorial k % (23 * k) = 0)) ∧
  (factorial 24 % (23 * 24) = 0) ∧
  (∀ k < 12, ¬(factorial k % (33 * k) = 0)) ∧
  (factorial 12 % (33 * 12) = 0) := by
  sorry

#check smallest_factorial_divisible_by_23m_and_33n

end NUMINAMATH_CALUDE_smallest_factorial_divisible_by_23m_and_33n_l3038_303821


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3038_303811

def f (x a b : ℝ) : ℝ := -|x - a| + b
def g (x c d : ℝ) : ℝ := |x - c| - d

theorem intersection_implies_sum (a b c d : ℝ) :
  f 1 a b = 4 ∧ g 1 c d = 4 ∧ f 7 a b = 2 ∧ g 7 c d = 2 → a + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3038_303811


namespace NUMINAMATH_CALUDE_max_pencils_buyable_l3038_303828

/-- Represents the number of pencils in a set -/
inductive PencilSet
| Large : PencilSet  -- 20 pencils
| Small : PencilSet  -- 5 pencils

/-- Represents the rebate percentage for a given set -/
def rebate_percentage (s : PencilSet) : ℚ :=
  match s with
  | PencilSet.Large => 25 / 100
  | PencilSet.Small => 10 / 100

/-- Represents the number of pencils in a given set -/
def pencils_in_set (s : PencilSet) : ℕ :=
  match s with
  | PencilSet.Large => 20
  | PencilSet.Small => 5

/-- The initial number of pencils Vasya can afford -/
def initial_pencils : ℕ := 30

/-- Theorem stating the maximum number of pencils Vasya can buy -/
theorem max_pencils_buyable :
  ∃ (large_sets small_sets : ℕ),
    large_sets * pencils_in_set PencilSet.Large +
    small_sets * pencils_in_set PencilSet.Small +
    ⌊large_sets * (rebate_percentage PencilSet.Large * pencils_in_set PencilSet.Large : ℚ)⌋ +
    ⌊small_sets * (rebate_percentage PencilSet.Small * pencils_in_set PencilSet.Small : ℚ)⌋ = 41 ∧
    large_sets * pencils_in_set PencilSet.Large +
    small_sets * pencils_in_set PencilSet.Small ≤ initial_pencils :=
by sorry

end NUMINAMATH_CALUDE_max_pencils_buyable_l3038_303828


namespace NUMINAMATH_CALUDE_students_not_enrolled_in_languages_l3038_303864

/-- Given a class with the following properties:
  * There are 150 students in total
  * 61 students are taking French
  * 32 students are taking German
  * 45 students are taking Spanish
  * 15 students are taking both French and German
  * 12 students are taking both French and Spanish
  * 10 students are taking both German and Spanish
  * 5 students are taking all three languages
  This theorem proves that the number of students not enrolled in any
  of these language courses is 44. -/
theorem students_not_enrolled_in_languages (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_and_german : ℕ) (french_and_spanish : ℕ) (german_and_spanish : ℕ) (all_three : ℕ)
  (h_total : total = 150)
  (h_french : french = 61)
  (h_german : german = 32)
  (h_spanish : spanish = 45)
  (h_french_and_german : french_and_german = 15)
  (h_french_and_spanish : french_and_spanish = 12)
  (h_german_and_spanish : german_and_spanish = 10)
  (h_all_three : all_three = 5) :
  total - (french + german + spanish - french_and_german - french_and_spanish - german_and_spanish + all_three) = 44 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_in_languages_l3038_303864


namespace NUMINAMATH_CALUDE_alberto_bjorn_difference_l3038_303855

/-- Represents a biker's travel distance over time --/
structure BikerTravel where
  miles : ℝ
  hours : ℝ

/-- Alberto's travel after 4 hours --/
def alberto : BikerTravel :=
  { miles := 60
  , hours := 4 }

/-- Bjorn's travel after 4 hours --/
def bjorn : BikerTravel :=
  { miles := 45
  , hours := 4 }

/-- The difference in miles traveled between two bikers --/
def mileDifference (a b : BikerTravel) : ℝ :=
  a.miles - b.miles

/-- Theorem stating the difference in miles traveled between Alberto and Bjorn --/
theorem alberto_bjorn_difference :
  mileDifference alberto bjorn = 15 := by
  sorry

end NUMINAMATH_CALUDE_alberto_bjorn_difference_l3038_303855


namespace NUMINAMATH_CALUDE_residue_calculation_l3038_303898

theorem residue_calculation : (240 * 15 - 21 * 9 + 6) % 18 = 15 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l3038_303898


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l3038_303894

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point K
def K : ℝ × ℝ := (-1, 0)

-- Define the property of a line passing through K
def line_through_K (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the intersection points A and B
def intersection_points (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_K m x₁ y₁ ∧ line_through_K m x₂ y₂ ∧
  y₁ ≠ y₂

-- Define point D as symmetric to A with respect to x-axis
def point_D (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, -y₁)

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- The main theorem
theorem fixed_point_theorem (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  m ≠ 0 →
  intersection_points m x₁ y₁ x₂ y₂ →
  ∃ (t : ℝ), t ∈ Set.Icc 0 1 ∧ 
    F.1 = (1 - t) * x₂ + t * (point_D x₁ y₁).1 ∧
    F.2 = (1 - t) * y₂ + t * (point_D x₁ y₁).2 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l3038_303894


namespace NUMINAMATH_CALUDE_calculation_proofs_l3038_303843

theorem calculation_proofs :
  (1.4 + (-0.2) + 0.6 + (-1.8) = 0) ∧
  ((-1/6 + 3/2 - 5/12) * (-48) = -44) ∧
  ((-1/3)^3 * (-3)^2 * (-1)^2011 = 1/3) ∧
  (-1^3 * (-5) / ((-3)^2 + 2 * (-5)) = -5) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l3038_303843


namespace NUMINAMATH_CALUDE_meal_preparation_assignments_l3038_303875

theorem meal_preparation_assignments (n : ℕ) (h : n = 6) :
  (n.choose 3) * ((n - 3).choose 1) * ((n - 4).choose 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_meal_preparation_assignments_l3038_303875


namespace NUMINAMATH_CALUDE_unemployment_rate_after_changes_l3038_303871

theorem unemployment_rate_after_changes (initial_unemployment : ℝ) : 
  initial_unemployment ≥ 0 ∧ initial_unemployment ≤ 100 →
  1.1 * initial_unemployment + 0.85 * (100 - initial_unemployment) = 100 →
  1.1 * initial_unemployment = 66 :=
by sorry

end NUMINAMATH_CALUDE_unemployment_rate_after_changes_l3038_303871


namespace NUMINAMATH_CALUDE_olympiad_team_formation_l3038_303814

theorem olympiad_team_formation (n : ℕ) (k : ℕ) (roles : ℕ) 
  (h1 : n = 20) 
  (h2 : k = 3) 
  (h3 : roles = 3) :
  (n.factorial / ((n - k).factorial * k.factorial)) * (k.factorial / (roles.factorial * (k - roles).factorial)) = 6840 :=
sorry

end NUMINAMATH_CALUDE_olympiad_team_formation_l3038_303814


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l3038_303888

def total_lamps : ℕ := 8
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 4
def lamps_turned_on : ℕ := 4

theorem lava_lamp_probability :
  let total_arrangements := Nat.choose total_lamps red_lamps
  let color_condition := Nat.choose (total_lamps - 4) (red_lamps - 2)
  let on_off_condition := Nat.choose (total_lamps - 2) (lamps_turned_on - 2)
  (color_condition * on_off_condition : ℚ) / (total_arrangements * total_arrangements) = 225 / 4900 := by
  sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l3038_303888


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l3038_303893

theorem circle_equation_m_range (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*(m - 3)*x + 2*y + 5 = 0 → ∃ h k r : ℝ, (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0) ↔
  (m > 5 ∨ m < 1) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l3038_303893


namespace NUMINAMATH_CALUDE_camel_cost_l3038_303830

/-- The cost of animals in a market --/
structure AnimalCosts where
  camel : ℕ
  horse : ℕ
  ox : ℕ
  elephant : ℕ

/-- The conditions of the animal costs problem --/
def animal_costs_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 150000

/-- The theorem stating that under the given conditions, a camel costs 6000 --/
theorem camel_cost (costs : AnimalCosts) : 
  animal_costs_conditions costs → costs.camel = 6000 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l3038_303830


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_ge_one_l3038_303815

/-- A function f(x) = ln x - ax is monotonically decreasing on (1, +∞) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, 1 < x → x < y → Real.log y - a * y < Real.log x - a * x

/-- If f(x) = ln x - ax is monotonically decreasing on (1, +∞), then a ≥ 1 -/
theorem monotone_decreasing_implies_a_ge_one (a : ℝ) :
  is_monotone_decreasing a → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_ge_one_l3038_303815


namespace NUMINAMATH_CALUDE_acid_solution_mixture_l3038_303804

/-- Given:
  n : ℝ, amount of initial solution in ounces
  y : ℝ, amount of added solution in ounces
  n > 30
  initial solution concentration is n%
  added solution concentration is 20%
  final solution concentration is (n-15)%
Prove: y = 15n / (n+35) -/
theorem acid_solution_mixture (n : ℝ) (y : ℝ) (h1 : n > 30) :
  (n * (n / 100) + y * (20 / 100)) / (n + y) = (n - 15) / 100 →
  y = 15 * n / (n + 35) := by
sorry

end NUMINAMATH_CALUDE_acid_solution_mixture_l3038_303804


namespace NUMINAMATH_CALUDE_symmetry_axis_of_quadratic_l3038_303840

/-- A quadratic function of the form y = (x + h)^2 has a symmetry axis of x = -h -/
theorem symmetry_axis_of_quadratic (h : ℝ) : 
  let f : ℝ → ℝ := λ x => (x + h)^2
  ∀ x : ℝ, f ((-h) - (x - (-h))) = f x := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_quadratic_l3038_303840


namespace NUMINAMATH_CALUDE_green_corner_plants_l3038_303853

theorem green_corner_plants (total_pots : ℕ) (green_lily_cost spider_plant_cost : ℕ) (total_budget : ℕ)
  (h1 : total_pots = 46)
  (h2 : green_lily_cost = 9)
  (h3 : spider_plant_cost = 6)
  (h4 : total_budget = 390) :
  ∃ (green_lily_pots spider_plant_pots : ℕ),
    green_lily_pots + spider_plant_pots = total_pots ∧
    green_lily_cost * green_lily_pots + spider_plant_cost * spider_plant_pots = total_budget ∧
    green_lily_pots = 38 ∧
    spider_plant_pots = 8 :=
by sorry

end NUMINAMATH_CALUDE_green_corner_plants_l3038_303853


namespace NUMINAMATH_CALUDE_share_calculation_l3038_303862

theorem share_calculation (total : ℝ) (a b c : ℝ) 
  (h_total : total = 700)
  (h_a_b : a = (1/2) * b)
  (h_b_c : b = (1/2) * c)
  (h_sum : a + b + c = total) :
  c = 400 := by
sorry

end NUMINAMATH_CALUDE_share_calculation_l3038_303862


namespace NUMINAMATH_CALUDE_probability_under_20_l3038_303807

theorem probability_under_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 130) (h2 : over_30 = 90) :
  (total - over_30 : ℚ) / total = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_under_20_l3038_303807


namespace NUMINAMATH_CALUDE_all_natural_numbers_reachable_l3038_303806

-- Define the operations
def f (n : ℕ) : ℕ := 10 * n

def g (n : ℕ) : ℕ := 10 * n + 4

def h (n : ℕ) : ℕ := n / 2

-- Define the set of reachable numbers
inductive Reachable : ℕ → Prop where
  | start : Reachable 4
  | apply_f {n : ℕ} : Reachable n → Reachable (f n)
  | apply_g {n : ℕ} : Reachable n → Reachable (g n)
  | apply_h {n : ℕ} : Even n → Reachable n → Reachable (h n)

-- Theorem statement
theorem all_natural_numbers_reachable : ∀ m : ℕ, Reachable m := by
  sorry

end NUMINAMATH_CALUDE_all_natural_numbers_reachable_l3038_303806


namespace NUMINAMATH_CALUDE_sales_balance_l3038_303817

/-- Represents the sales increase of product C as a percentage -/
def sales_increase_C : ℝ := 0.3

/-- Represents the proportion of total sales from product C last year -/
def last_year_C_proportion : ℝ := 0.4

/-- Represents the decrease in sales for products A and B -/
def sales_decrease_AB : ℝ := 0.2

/-- Represents the proportion of total sales from products A and B last year -/
def last_year_AB_proportion : ℝ := 1 - last_year_C_proportion

theorem sales_balance :
  last_year_C_proportion * (1 + sales_increase_C) + 
  last_year_AB_proportion * (1 - sales_decrease_AB) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sales_balance_l3038_303817


namespace NUMINAMATH_CALUDE_basketball_preference_theorem_l3038_303837

/-- Represents the school population and basketball preferences -/
structure School where
  total_students : ℕ
  male_ratio : ℚ
  female_ratio : ℚ
  male_basketball_ratio : ℚ
  female_basketball_ratio : ℚ

/-- Calculate the percentage of students who do not like basketball -/
def percentage_not_liking_basketball (s : School) : ℚ :=
  let male_students := s.total_students * s.male_ratio / (s.male_ratio + s.female_ratio)
  let female_students := s.total_students * s.female_ratio / (s.male_ratio + s.female_ratio)
  let male_liking_basketball := male_students * s.male_basketball_ratio
  let female_liking_basketball := female_students * s.female_basketball_ratio
  let total_not_liking := s.total_students - (male_liking_basketball + female_liking_basketball)
  total_not_liking / s.total_students * 100

/-- The main theorem to prove -/
theorem basketball_preference_theorem (s : School) 
  (h1 : s.total_students = 1000)
  (h2 : s.male_ratio = 3)
  (h3 : s.female_ratio = 2)
  (h4 : s.male_basketball_ratio = 2/3)
  (h5 : s.female_basketball_ratio = 1/5) :
  percentage_not_liking_basketball s = 52 := by
  sorry


end NUMINAMATH_CALUDE_basketball_preference_theorem_l3038_303837


namespace NUMINAMATH_CALUDE_truck_travel_distance_l3038_303863

/-- Given a truck that travels 150 miles on 5 gallons of diesel,
    prove that it can travel 210 miles on 7 gallons of diesel,
    assuming a constant rate of travel. -/
theorem truck_travel_distance 
  (initial_distance : ℝ) 
  (initial_fuel : ℝ) 
  (new_fuel : ℝ) 
  (h1 : initial_distance = 150) 
  (h2 : initial_fuel = 5) 
  (h3 : new_fuel = 7) :
  (initial_distance / initial_fuel) * new_fuel = 210 :=
by sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l3038_303863


namespace NUMINAMATH_CALUDE_black_or_white_probability_l3038_303890

/-- The probability of drawing a red ball from the box -/
def prob_red : ℝ := 0.45

/-- The probability of drawing a white ball from the box -/
def prob_white : ℝ := 0.25

/-- The probability of drawing either a black ball or a white ball from the box -/
def prob_black_or_white : ℝ := 1 - prob_red

theorem black_or_white_probability : prob_black_or_white = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_black_or_white_probability_l3038_303890


namespace NUMINAMATH_CALUDE_compound_composition_l3038_303800

/-- Atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1

/-- Atomic weight of chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.5

/-- Atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 68

/-- Number of oxygen atoms in the compound -/
def n : ℕ := 2

theorem compound_composition :
  molecular_weight = atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O :=
sorry

end NUMINAMATH_CALUDE_compound_composition_l3038_303800


namespace NUMINAMATH_CALUDE_certain_number_proof_l3038_303884

theorem certain_number_proof (N : ℚ) : 
  (5 / 6 : ℚ) * N - (5 / 16 : ℚ) * N = 200 → N = 384 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3038_303884


namespace NUMINAMATH_CALUDE_power_product_equals_78125_l3038_303869

theorem power_product_equals_78125 (a : ℕ) (h : a = 5) : a^3 * a^4 = 78125 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_78125_l3038_303869


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3038_303892

theorem polynomial_division_remainder :
  ∃ (q r : Polynomial ℝ),
    x^4 + 5 = (x^2 - 4*x + 7) * q + r ∧
    r.degree < (x^2 - 4*x + 7).degree ∧
    r = 8*x - 58 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3038_303892


namespace NUMINAMATH_CALUDE_specific_ellipse_semi_minor_axis_l3038_303825

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  semi_major_endpoint : ℝ × ℝ

/-- Calculates the semi-minor axis of an ellipse -/
def semi_minor_axis (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the semi-minor axis of the specific ellipse is √21 -/
theorem specific_ellipse_semi_minor_axis :
  let e : Ellipse := {
    center := (0, 0),
    focus := (0, -2),
    semi_major_endpoint := (0, 5)
  }
  semi_minor_axis e = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_semi_minor_axis_l3038_303825


namespace NUMINAMATH_CALUDE_some_number_value_l3038_303891

theorem some_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = x * 25 * 45 * 49) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3038_303891


namespace NUMINAMATH_CALUDE_triangle_tangent_product_range_l3038_303851

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    satisfying a^2 + b^2 + √2ab = c^2, prove that 0 < tan A * tan (2*B) < 1/2 -/
theorem triangle_tangent_product_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + b^2 + Real.sqrt 2 * a * b = c^2 →
  0 < Real.tan A * Real.tan (2 * B) ∧ Real.tan A * Real.tan (2 * B) < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_range_l3038_303851


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3038_303886

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) (h4 : a 4 = 5) :
  a 1 * a 7 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3038_303886


namespace NUMINAMATH_CALUDE_no_single_non_divisible_l3038_303823

/-- Represents a 5x5 table of non-zero digits -/
def Table := Fin 5 → Fin 5 → Fin 9

/-- Checks if a number is divisible by 3 -/
def isDivisibleBy3 (n : ℕ) : Prop := n % 3 = 0

/-- Sums the digits in a row or column -/
def sumDigits (digits : Fin 5 → Fin 9) : ℕ :=
  (Finset.univ.sum fun i => (digits i).val) + 5

/-- Theorem stating the impossibility of having exactly one number not divisible by 3 -/
theorem no_single_non_divisible (t : Table) : 
  ¬ (∃! n : Fin 10, ¬ isDivisibleBy3 (sumDigits (fun i => 
    if n.val < 5 then t i n.val else t n.val (i - 5)))) := by
  sorry

end NUMINAMATH_CALUDE_no_single_non_divisible_l3038_303823


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3038_303827

/-- Calculate the rate of simple interest given principal, time, and interest amount -/
theorem simple_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (interest : ℝ) 
  (h1 : principal = 20000)
  (h2 : time = 3)
  (h3 : interest = 7200) : 
  (interest * 100) / (principal * time) = 12 := by
  sorry

#check simple_interest_rate

end NUMINAMATH_CALUDE_simple_interest_rate_l3038_303827


namespace NUMINAMATH_CALUDE_best_approx_sqrt3_l3038_303819

def best_rational_approx (n : ℕ) (x : ℝ) : ℚ :=
  sorry

theorem best_approx_sqrt3 :
  best_rational_approx 15 (Real.sqrt 3) = 26 / 15 := by
  sorry

end NUMINAMATH_CALUDE_best_approx_sqrt3_l3038_303819


namespace NUMINAMATH_CALUDE_probability_prime_8_sided_die_l3038_303849

-- Define a fair 8-sided die
def fair_8_sided_die : Finset ℕ := Finset.range 8

-- Define the set of prime numbers from 1 to 8
def primes_1_to_8 : Finset ℕ := {2, 3, 5, 7}

-- Theorem: The probability of rolling a prime number on a fair 8-sided die is 1/2
theorem probability_prime_8_sided_die :
  (Finset.card primes_1_to_8 : ℚ) / (Finset.card fair_8_sided_die : ℚ) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_probability_prime_8_sided_die_l3038_303849


namespace NUMINAMATH_CALUDE_investment_result_l3038_303878

/-- Calculates the future value of an investment with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that an investment of $4000 at 10% annual compound interest for 2 years results in $4840 -/
theorem investment_result : compound_interest 4000 0.1 2 = 4840 := by
  sorry

end NUMINAMATH_CALUDE_investment_result_l3038_303878


namespace NUMINAMATH_CALUDE_cryptarithmetic_problem_l3038_303816

theorem cryptarithmetic_problem (A B C : ℕ) : 
  A < 10 → B < 10 → C < 10 →  -- Single-digit integers
  A ≠ B → A ≠ C → B ≠ C →     -- Unique digits
  A * B = 24 →                -- First equation
  A - C = 4 →                 -- Second equation
  C = 0 :=                    -- Conclusion
by
  sorry

end NUMINAMATH_CALUDE_cryptarithmetic_problem_l3038_303816


namespace NUMINAMATH_CALUDE_binomial_12_11_l3038_303895

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l3038_303895


namespace NUMINAMATH_CALUDE_arithmetic_problem_l3038_303810

theorem arithmetic_problem : 4 * (8 - 3)^2 - 2 * 7 = 86 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l3038_303810


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l3038_303832

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the theorem
theorem perpendicular_transitivity
  (m n : Line) (α β : Plane)
  (h1 : perpendicular m β)
  (h2 : perpendicular n β)
  (h3 : perpendicular n α) :
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l3038_303832


namespace NUMINAMATH_CALUDE_gumball_pigeonhole_min_gumballs_for_five_same_color_l3038_303833

theorem gumball_pigeonhole : ∀ (draw : ℕ),
  (∃ (color : Fin 4), (λ i => [10, 8, 9, 7].get i) color ≥ 5) →
  draw ≥ 17 :=
by
  sorry

theorem min_gumballs_for_five_same_color :
  ∃ (draw : ℕ), draw = 17 ∧
  (∀ (smaller : ℕ), smaller < draw →
    ¬∃ (color : Fin 4), (λ i => [10, 8, 9, 7].get i) color ≥ 5) ∧
  (∃ (color : Fin 4), (λ i => [10, 8, 9, 7].get i) color ≥ 5) :=
by
  sorry

end NUMINAMATH_CALUDE_gumball_pigeonhole_min_gumballs_for_five_same_color_l3038_303833


namespace NUMINAMATH_CALUDE_organic_egg_tray_price_l3038_303805

/-- The price of a tray of organic eggs -/
def tray_price (individual_price : ℚ) (tray_size : ℕ) (savings_per_egg : ℚ) : ℚ :=
  (individual_price - savings_per_egg) * tray_size / 100

/-- Proof that the price of a tray of 30 organic eggs is $12 -/
theorem organic_egg_tray_price :
  let individual_price : ℚ := 50
  let tray_size : ℕ := 30
  let savings_per_egg : ℚ := 10
  tray_price individual_price tray_size savings_per_egg = 12 := by sorry

end NUMINAMATH_CALUDE_organic_egg_tray_price_l3038_303805


namespace NUMINAMATH_CALUDE_student_preferences_l3038_303887

/-- In a class of 30 students, prove that the sum of students who like maths and history is 15,
    given the distribution of student preferences. -/
theorem student_preferences (total : ℕ) (maths_ratio science_ratio history_ratio : ℚ) : 
  total = 30 ∧ 
  maths_ratio = 3/10 ∧ 
  science_ratio = 1/4 ∧ 
  history_ratio = 2/5 → 
  ∃ (maths science history literature : ℕ),
    maths = ⌊maths_ratio * total⌋ ∧
    science = ⌊science_ratio * (total - maths)⌋ ∧
    history = ⌊history_ratio * (total - maths - science)⌋ ∧
    literature = total - maths - science - history ∧
    maths + history = 15 :=
by sorry


end NUMINAMATH_CALUDE_student_preferences_l3038_303887


namespace NUMINAMATH_CALUDE_periodic_function_l3038_303818

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (10 - x) = 4

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : satisfies_condition f) : 
  is_periodic f 20 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_l3038_303818
