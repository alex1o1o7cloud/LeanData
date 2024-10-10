import Mathlib

namespace total_jeans_is_five_l1566_156665

/-- The number of Fox jeans purchased -/
def fox_jeans : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_jeans : ℕ := 2

/-- The total number of jeans purchased -/
def total_jeans : ℕ := fox_jeans + pony_jeans

theorem total_jeans_is_five : total_jeans = 5 := by sorry

end total_jeans_is_five_l1566_156665


namespace parabola_directrix_l1566_156695

theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 → (∃ k : ℝ, y = 1 ↔ x^2 = 1 / (4 * k))) → 
  a = -1/4 := by sorry

end parabola_directrix_l1566_156695


namespace g_sum_property_l1566_156697

def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^8 + e * x^6 - f * x^4 + 5

theorem g_sum_property (d e f : ℝ) :
  g d e f 20 = 7 → g d e f 20 + g d e f (-20) = 14 := by
  sorry

end g_sum_property_l1566_156697


namespace allan_has_one_more_balloon_l1566_156629

/-- Given the number of balloons Allan and Jake have, prove that Allan has one more balloon than Jake. -/
theorem allan_has_one_more_balloon (allan_balloons jake_initial_balloons jake_bought_balloons : ℕ) 
  (h1 : allan_balloons = 6)
  (h2 : jake_initial_balloons = 2)
  (h3 : jake_bought_balloons = 3) :
  allan_balloons - (jake_initial_balloons + jake_bought_balloons) = 1 := by
  sorry

end allan_has_one_more_balloon_l1566_156629


namespace five_fourths_of_twelve_fifths_l1566_156603

theorem five_fourths_of_twelve_fifths (x : ℚ) : x = 12 / 5 → (5 / 4) * x = 3 := by
  sorry

end five_fourths_of_twelve_fifths_l1566_156603


namespace sarah_sock_purchase_l1566_156674

/-- Represents the number of pairs of socks at each price point --/
structure SockCounts where
  two_dollar : ℕ
  four_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the given sock counts satisfy the problem conditions --/
def is_valid_solution (s : SockCounts) : Prop :=
  s.two_dollar + s.four_dollar + s.five_dollar = 15 ∧
  2 * s.two_dollar + 4 * s.four_dollar + 5 * s.five_dollar = 45 ∧
  s.two_dollar ≥ 1 ∧ s.four_dollar ≥ 1 ∧ s.five_dollar ≥ 1

theorem sarah_sock_purchase :
  ∃ (s : SockCounts), is_valid_solution s ∧ (s.two_dollar = 8 ∨ s.two_dollar = 9) :=
sorry

end sarah_sock_purchase_l1566_156674


namespace man_son_age_ratio_l1566_156604

theorem man_son_age_ratio :
  ∀ (man_age son_age : ℕ),
    man_age = 36 →
    son_age = 12 →
    man_age + 12 = 2 * (son_age + 12) →
    man_age / son_age = 3 :=
by
  sorry

end man_son_age_ratio_l1566_156604


namespace major_axis_length_is_8_l1566_156647

/-- The length of the major axis of an ellipse formed by intersecting a plane with a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The length of the major axis is 8 when a plane intersects a right circular cylinder with radius 2, forming an ellipse where the major axis is double the minor axis -/
theorem major_axis_length_is_8 :
  major_axis_length 2 2 = 8 := by
  sorry

end major_axis_length_is_8_l1566_156647


namespace store_owner_uniforms_l1566_156672

theorem store_owner_uniforms :
  ∃ (U : ℕ), 
    (U > 0) ∧ 
    (∃ (E : ℕ), U + 1 = 2 * E) ∧ 
    (∀ (V : ℕ), V < U → ¬(∃ (F : ℕ), V + 1 = 2 * F)) ∧
    (U = 3) := by
  sorry

end store_owner_uniforms_l1566_156672


namespace largest_gcd_value_l1566_156644

theorem largest_gcd_value (n : ℕ) : 
  ∃ (m : ℕ), (∀ k : ℕ, Nat.gcd (k^2 + 3) ((k + 1)^2 + 3) ≤ m) ∧ 
             (Nat.gcd (n^2 + 3) ((n + 1)^2 + 3) = m) ∧
             m = 13 := by
  sorry

end largest_gcd_value_l1566_156644


namespace cone_height_l1566_156608

theorem cone_height (r l h : ℝ) : 
  r = 1 → l = 4 → l^2 = r^2 + h^2 → h = Real.sqrt 15 := by sorry

end cone_height_l1566_156608


namespace laura_weekly_mileage_l1566_156607

/-- Represents the total miles driven by Laura in a week -/
def total_miles_per_week (
  house_school_round_trip : ℕ)
  (supermarket_extra_distance : ℕ)
  (gym_distance : ℕ)
  (friend_distance : ℕ)
  (workplace_distance : ℕ)
  (school_days : ℕ)
  (supermarket_trips : ℕ)
  (gym_trips : ℕ)
  (friend_trips : ℕ) : ℕ :=
  -- Weekday trips (work and school)
  (workplace_distance + (house_school_round_trip / 2 - workplace_distance) + (house_school_round_trip / 2)) * school_days +
  -- Supermarket trips
  ((house_school_round_trip / 2 + supermarket_extra_distance) * 2) * supermarket_trips +
  -- Gym trips
  (gym_distance * 2) * gym_trips +
  -- Friend's house trips
  (friend_distance * 2) * friend_trips

/-- Theorem stating that Laura drives 234 miles per week -/
theorem laura_weekly_mileage :
  total_miles_per_week 20 10 5 12 8 5 2 3 1 = 234 := by
  sorry

end laura_weekly_mileage_l1566_156607


namespace line_parallel_to_plane_perpendicular_to_two_planes_are_parallel_l1566_156650

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)

-- Statement ②
theorem line_parallel_to_plane 
  (l : Line) (α : Plane) :
  parallel_plane l α → 
  ∃ (S : Set Line), (∀ m ∈ S, in_plane m α ∧ parallel l m) ∧ Set.Infinite S :=
sorry

-- Statement ④
theorem perpendicular_to_two_planes_are_parallel 
  (m : Line) (α β : Plane) :
  perpendicular_plane m α → perpendicular_plane m β → parallel_planes α β :=
sorry

end line_parallel_to_plane_perpendicular_to_two_planes_are_parallel_l1566_156650


namespace hiker_walking_problem_l1566_156602

/-- A hiker's walking problem over three days -/
theorem hiker_walking_problem 
  (day1_distance : ℝ) 
  (day1_speed : ℝ) 
  (day2_speed_increase : ℝ) 
  (day3_speed : ℝ) 
  (day3_time : ℝ) 
  (total_distance : ℝ) 
  (h1 : day1_distance = 18) 
  (h2 : day1_speed = 3) 
  (h3 : day2_speed_increase = 1) 
  (h4 : day3_speed = 5) 
  (h5 : day3_time = 6) 
  (h6 : total_distance = 68) :
  day1_distance / day1_speed - 
  (total_distance - day1_distance - day3_speed * day3_time) / (day1_speed + day2_speed_increase) = 1 := by
  sorry

end hiker_walking_problem_l1566_156602


namespace correct_average_l1566_156646

/-- Given 10 numbers with an initial average of 14, where one number 36 was incorrectly read as 26, prove that the correct average is 15. -/
theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 14 →
  incorrect_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 15 :=
by sorry

end correct_average_l1566_156646


namespace fraction_equality_l1566_156615

theorem fraction_equality : (1 + 5) / (3 + 5) = 3 / 4 := by
  sorry

end fraction_equality_l1566_156615


namespace zero_in_interval_l1566_156616

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 4 ∧ f c = 0 := by sorry

end zero_in_interval_l1566_156616


namespace inheritance_problem_l1566_156610

theorem inheritance_problem (total_inheritance : ℕ) (additional_amount : ℕ) : 
  total_inheritance = 46800 →
  additional_amount = 1950 →
  ∃ (original_children : ℕ),
    original_children > 2 ∧
    (total_inheritance / original_children + additional_amount = total_inheritance / (original_children - 2)) ∧
    original_children = 8 := by
  sorry

end inheritance_problem_l1566_156610


namespace skier_total_time_l1566_156654

theorem skier_total_time (x : ℝ) (t₁ t₂ t₃ : ℝ) 
  (h1 : t₁ + t₂ = 40.5)
  (h2 : t₂ + t₃ = 37.5)
  (h3 : x / t₂ = (2 * x) / (t₁ + t₃))
  (h4 : x > 0) :
  t₁ + t₂ + t₃ = 58.5 := by
sorry

end skier_total_time_l1566_156654


namespace tennis_players_count_l1566_156605

theorem tennis_players_count (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 40)
  (h2 : badminton = 20)
  (h3 : neither = 5)
  (h4 : both = 3)
  : ∃ tennis : ℕ, tennis = 18 ∧ 
    total = badminton + tennis - both + neither :=
by sorry

end tennis_players_count_l1566_156605


namespace notebook_ratio_l1566_156677

theorem notebook_ratio (total_students : ℕ) (total_notebooks : ℕ) 
  (h1 : total_students = 28)
  (h2 : total_notebooks = 112)
  (h3 : ∃ (x y : ℕ), x + y = total_students ∧ y = total_students / 2 ∧ 5 * x + 3 * y = total_notebooks) :
  ∃ (x y : ℕ), x = y ∧ x + y = total_students ∧ 5 * x + 3 * y = total_notebooks := by
  sorry

end notebook_ratio_l1566_156677


namespace ratio_a_to_b_l1566_156625

def arithmetic_sequence (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | n+1 => arithmetic_sequence a d n + d

theorem ratio_a_to_b (a d : ℝ) :
  let b := a + 3 * d
  (arithmetic_sequence a d 0 = a) ∧
  (arithmetic_sequence a d 1 = a + 2*d) ∧
  (arithmetic_sequence a d 2 = a + 3*d) ∧
  (arithmetic_sequence a d 3 = a + 5*d) →
  a / b = 1 / 4 := by
sorry

end ratio_a_to_b_l1566_156625


namespace inequality_preservation_l1566_156685

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end inequality_preservation_l1566_156685


namespace shaded_area_between_circles_l1566_156631

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) : 
  π * r₂^2 - π * r₁^2 = 48 * π := by
  sorry

end shaded_area_between_circles_l1566_156631


namespace mikes_bills_l1566_156620

theorem mikes_bills (total_amount : ℕ) (bill_denomination : ℕ) (h1 : total_amount = 45) (h2 : bill_denomination = 5) :
  total_amount / bill_denomination = 9 := by
  sorry

end mikes_bills_l1566_156620


namespace partial_fraction_decomposition_l1566_156660

theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ), ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 →
    (-x^2 + 4*x - 5) / (x^3 + x) = A / x + (B*x + C) / (x^2 + 1) ∧
    A = -5 ∧ B = 4 ∧ C = 4 :=
by sorry

end partial_fraction_decomposition_l1566_156660


namespace cubic_difference_equality_l1566_156656

theorem cubic_difference_equality (x y : ℝ) : 
  x^2 = 7 + 4 * Real.sqrt 3 ∧ 
  y^2 = 7 - 4 * Real.sqrt 3 → 
  x^3 / y - y^3 / x = 112 * Real.sqrt 3 := by
sorry

end cubic_difference_equality_l1566_156656


namespace sum_of_first_few_primes_equals_41_l1566_156661

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a unique positive integer n such that the sum of the first n prime numbers equals 41, and that n = 6 -/
theorem sum_of_first_few_primes_equals_41 :
  ∃! n : ℕ, n > 0 ∧ sumFirstNPrimes n = 41 ∧ n = 6 := by sorry

end sum_of_first_few_primes_equals_41_l1566_156661


namespace friday_thirteenth_most_common_l1566_156651

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month in the Gregorian calendar -/
structure Month where
  startDay : DayOfWeek
  length : Nat

/-- Represents a year in the Gregorian calendar -/
structure Year where
  isLeap : Bool
  months : List Month

/-- Calculates the day of week for the 13th of a given month -/
def thirteenthDayOfMonth (m : Month) : DayOfWeek :=
  sorry

/-- Counts the occurrences of each day as the 13th in a 400-year cycle -/
def countThirteenths (years : List Year) : DayOfWeek → Nat :=
  sorry

/-- The Gregorian calendar repeats every 400 years -/
def gregorianCycle : List Year :=
  sorry

/-- Main theorem: Friday is the most common day for the 13th of a month -/
theorem friday_thirteenth_most_common :
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Friday →
    countThirteenths gregorianCycle DayOfWeek.Friday > countThirteenths gregorianCycle d :=
  sorry

end friday_thirteenth_most_common_l1566_156651


namespace cube_sum_inequality_l1566_156642

theorem cube_sum_inequality (a b c : ℤ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + Real.sqrt (3 * (a * b + b * c + c * a + 1)) :=
sorry

end cube_sum_inequality_l1566_156642


namespace selene_sandwich_count_l1566_156645

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℕ := 2

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℕ := 2

/-- The cost of a hotdog in dollars -/
def hotdog_cost : ℕ := 1

/-- The cost of a can of fruit juice in dollars -/
def juice_cost : ℕ := 2

/-- The number of hamburgers Tanya buys -/
def tanya_hamburgers : ℕ := 2

/-- The number of cans of fruit juice Tanya buys -/
def tanya_juice : ℕ := 2

/-- The total amount spent by Selene and Tanya in dollars -/
def total_spent : ℕ := 16

/-- The number of sandwiches Selene bought -/
def selene_sandwiches : ℕ := 3

theorem selene_sandwich_count :
  ∃ (x : ℕ), x * sandwich_cost + juice_cost + 
  tanya_hamburgers * hamburger_cost + tanya_juice * juice_cost = total_spent ∧
  x = selene_sandwiches :=
by sorry

end selene_sandwich_count_l1566_156645


namespace cuboid_surface_area_example_l1566_156671

/-- The surface area of a cuboid with given dimensions. -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + breadth * height + length * height)

/-- Theorem: The surface area of a cuboid with length 15, breadth 10, and height 16 is 1100. -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 15 10 16 = 1100 := by
  sorry

end cuboid_surface_area_example_l1566_156671


namespace remaining_segment_length_l1566_156626

/-- Represents an equilateral triangle with segments drawn from vertices to opposite sides. -/
structure SegmentedEquilateralTriangle where
  /-- Length of the first segment on one side -/
  a : ℝ
  /-- Length of the second segment on one side -/
  b : ℝ
  /-- Length of the third segment on one side -/
  c : ℝ
  /-- Length of the shortest segment on another side -/
  d : ℝ
  /-- Length of the segment adjacent to the shortest segment -/
  e : ℝ
  /-- Assumption that the triangle is equilateral and segments form a complete side -/
  side_length : a + b + c = d + e + (a + b + c - (d + e))

/-- Theorem stating that the remaining segment length is 4 cm given the conditions -/
theorem remaining_segment_length
  (triangle : SegmentedEquilateralTriangle)
  (h1 : triangle.a = 5)
  (h2 : triangle.b = 10)
  (h3 : triangle.c = 2)
  (h4 : triangle.d = 1.5)
  (h5 : triangle.e = 11.5) :
  triangle.a + triangle.b + triangle.c - (triangle.d + triangle.e) = 4 :=
by
  sorry

end remaining_segment_length_l1566_156626


namespace square_side_length_l1566_156663

theorem square_side_length : ∃ (X : ℝ), X = 2.6 ∧ 
  (∃ (A B C D : ℝ × ℝ),
    -- Four points inside the square
    (0 < A.1 ∧ A.1 < X) ∧ (0 < A.2 ∧ A.2 < X) ∧
    (0 < B.1 ∧ B.1 < X) ∧ (0 < B.2 ∧ B.2 < X) ∧
    (0 < C.1 ∧ C.1 < X) ∧ (0 < C.2 ∧ C.2 < X) ∧
    (0 < D.1 ∧ D.1 < X) ∧ (0 < D.2 ∧ D.2 < X) ∧
    -- Nine segments of length 1
    (A.1 - 0)^2 + (A.2 - 0)^2 = 1 ∧
    (B.1 - X)^2 + (B.2 - X)^2 = 1 ∧
    (C.1 - 0)^2 + (C.2 - X)^2 = 1 ∧
    (D.1 - X)^2 + (D.2 - 0)^2 = 1 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = 1 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = 1 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = 1 ∧
    -- Perpendicular segments
    A.1 = 0 ∧ B.1 = X ∧ C.2 = X ∧ D.2 = 0 ∧
    -- Distance conditions
    A.1 = (X - 1) / 2 ∧
    X - B.1 = 1) :=
by sorry

end square_side_length_l1566_156663


namespace function_properties_l1566_156635

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ x y, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)) :
  (f 0 = 1) ∧ (∀ x, f (-x) = f x) := by
sorry

end function_properties_l1566_156635


namespace positive_real_solution_of_equation_l1566_156637

theorem positive_real_solution_of_equation : 
  ∃! (x : ℝ), x > 0 ∧ (x - 6) / 11 = 6 / (x - 11) ∧ x = 17 := by
  sorry

end positive_real_solution_of_equation_l1566_156637


namespace coin_flip_probability_l1566_156682

theorem coin_flip_probability (p : ℝ) : 
  (p ≥ 0 ∧ p ≤ 1) →  -- probability is between 0 and 1
  (∀ (n : ℕ), n > 0 → p = 1 - p) →  -- equal probability of heads and tails
  (3 : ℝ) * p^2 * (1 - p) = (3 / 8 : ℝ) →  -- probability of 2 heads in 3 flips is 0.375
  p = (1 / 2 : ℝ) := by
sorry

end coin_flip_probability_l1566_156682


namespace shortest_chord_length_max_triangle_area_l1566_156691

-- Define the circle and point A
def circle_radius : ℝ := 1
def distance_OA (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem for the shortest chord length
theorem shortest_chord_length (a : ℝ) (h : distance_OA a) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt (1 - a^2) ∧
  ∀ (other_chord : ℝ), other_chord ≥ chord_length :=
sorry

-- Theorem for the maximum area of triangle OMN
theorem max_triangle_area (a : ℝ) (h : distance_OA a) :
  ∃ (max_area : ℝ),
    (a ≥ Real.sqrt 2 / 2 → max_area = 1 / 2) ∧
    (a < Real.sqrt 2 / 2 → max_area = a * Real.sqrt (1 - a^2)) :=
sorry

end shortest_chord_length_max_triangle_area_l1566_156691


namespace unique_valid_prism_l1566_156676

/-- A right rectangular prism with integer side lengths -/
structure RectPrism where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a ≤ b
  h2 : b ≤ c

/-- Predicate for a valid division of a prism -/
def validDivision (p : RectPrism) : Prop :=
  ∃ (k : ℚ), 0 < k ∧ k < 1 ∧
  ((k * p.a.val = p.a.val ∧ k * p.b.val = p.a.val) ∨
   (k * p.b.val = p.b.val ∧ k * p.c.val = p.b.val) ∨
   (k * p.c.val = p.c.val ∧ k * p.a.val = p.c.val))

theorem unique_valid_prism :
  ∃! (p : RectPrism), p.b = 101 ∧ validDivision p :=
sorry

end unique_valid_prism_l1566_156676


namespace ellipse_equation_from_hyperbola_l1566_156619

/-- Given a hyperbola with equation 3x^2 - y^2 = 3, prove that an ellipse with the same foci
    and reciprocal eccentricity has the equation x^2/16 + y^2/12 = 1 -/
theorem ellipse_equation_from_hyperbola (x y : ℝ) :
  (3 * x^2 - y^2 = 3) →
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧
    a^2 - b^2 = 4 ∧ 2 / a = 1 / 2) →
  x^2 / 16 + y^2 / 12 = 1 := by
  sorry

end ellipse_equation_from_hyperbola_l1566_156619


namespace smaller_cubes_count_l1566_156680

theorem smaller_cubes_count (large_volume : ℝ) (small_volume : ℝ) (surface_area_diff : ℝ) :
  large_volume = 125 →
  small_volume = 1 →
  surface_area_diff = 600 →
  (((6 * small_volume^(2/3)) * (large_volume / small_volume)) - (6 * large_volume^(2/3))) = surface_area_diff →
  (large_volume / small_volume) = 125 := by
  sorry

end smaller_cubes_count_l1566_156680


namespace union_of_A_and_B_l1566_156662

-- Define set A
def A : Set ℝ := {x | x^2 - x = 0}

-- Define set B
def B : Set ℝ := {-1, 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end union_of_A_and_B_l1566_156662


namespace inequality_relationship_l1566_156679

theorem inequality_relationship :
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ x ≥ -1) := by
  sorry

end inequality_relationship_l1566_156679


namespace ratio_problem_l1566_156699

theorem ratio_problem (x : ℝ) : x / 10 = 17.5 / 1 → x = 175 := by
  sorry

end ratio_problem_l1566_156699


namespace one_root_quadratic_l1566_156628

theorem one_root_quadratic (a : ℤ) : 
  (∃! x : ℝ, x ∈ Set.Icc 1 8 ∧ (x - a - 4)^2 + 2*x - 2*a - 16 = 0) ↔ 
  (a ∈ Set.Icc (-5) 0 ∨ a ∈ Set.Icc 3 8) :=
sorry

end one_root_quadratic_l1566_156628


namespace quadratic_equal_roots_l1566_156600

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x - k = 0 ∧ 
   ∀ y : ℝ, y^2 + 3*y - k = 0 → y = x) →
  k = -9/4 := by
sorry

end quadratic_equal_roots_l1566_156600


namespace complex_modulus_l1566_156624

theorem complex_modulus (x y : ℝ) (z : ℂ) : 
  z = x + y * Complex.I → 
  (1/2 * x - y : ℂ) + (x + y) * Complex.I = 3 * Complex.I → 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l1566_156624


namespace ounces_per_can_l1566_156601

/-- Represents the number of ounces in a cup of chickpeas -/
def ounces_per_cup : ℕ := 6

/-- Represents the number of cups needed for one serving of hummus -/
def cups_per_serving : ℕ := 1

/-- Represents the number of servings Thomas wants to make -/
def total_servings : ℕ := 20

/-- Represents the number of cans Thomas needs to buy -/
def cans_needed : ℕ := 8

/-- Theorem stating the number of ounces in each can of chickpeas -/
theorem ounces_per_can : 
  (total_servings * cups_per_serving * ounces_per_cup) / cans_needed = 15 := by
  sorry

end ounces_per_can_l1566_156601


namespace union_of_A_and_B_l1566_156627

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {3, 5, 7, 8}

theorem union_of_A_and_B : A ∪ B = {3, 4, 5, 6, 7, 8} := by
  sorry

end union_of_A_and_B_l1566_156627


namespace complex_sum_powers_of_i_l1566_156618

theorem complex_sum_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ i + i^2 + i^3 + i^4 = 0 := by sorry

end complex_sum_powers_of_i_l1566_156618


namespace fathers_age_problem_l1566_156692

/-- The father's age problem -/
theorem fathers_age_problem (man_age father_age : ℕ) : 
  man_age = (2 * father_age) / 5 →
  man_age + 12 = (father_age + 12) / 2 →
  father_age = 60 := by
  sorry

end fathers_age_problem_l1566_156692


namespace length_BC_is_sqrt_13_l1566_156613

/-- The cosine theorem for a triangle ABC -/
def cosine_theorem (a b c : ℝ) (A : ℝ) : Prop :=
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos A)

/-- Triangle ABC with given side lengths and angle -/
structure Triangle :=
  (AB : ℝ)
  (AC : ℝ)
  (angle_A : ℝ)
  (h_AB_pos : AB > 0)
  (h_AC_pos : AC > 0)
  (h_angle_A_pos : angle_A > 0)
  (h_angle_A_lt_pi : angle_A < π)

theorem length_BC_is_sqrt_13 (t : Triangle) 
  (h_AB : t.AB = 3)
  (h_AC : t.AC = 4)
  (h_angle_A : t.angle_A = π/3) :
  ∃ BC : ℝ, BC > 0 ∧ BC^2 = 13 ∧ cosine_theorem t.AB t.AC BC t.angle_A :=
sorry

end length_BC_is_sqrt_13_l1566_156613


namespace birthday_cars_count_l1566_156688

-- Define the initial number of cars
def initial_cars : ℕ := 14

-- Define the number of cars bought
def bought_cars : ℕ := 28

-- Define the number of cars given away
def given_away_cars : ℕ := 8 + 3

-- Define the final number of cars
def final_cars : ℕ := 43

-- Theorem to prove
theorem birthday_cars_count :
  ∃ (birthday_cars : ℕ), 
    initial_cars + bought_cars + birthday_cars - given_away_cars = final_cars ∧
    birthday_cars = 12 := by
  sorry

end birthday_cars_count_l1566_156688


namespace rectangle_area_l1566_156606

theorem rectangle_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (a^2 * b) / (b^2 * a) = 5/8 →
  (a + 6) * (b + 6) - a * b = 114 →
  a * b = 40 := by sorry

end rectangle_area_l1566_156606


namespace cloth_cost_price_l1566_156617

/-- Given a trader selling cloth, calculate the cost price per metre. -/
theorem cloth_cost_price
  (total_metres : ℕ)
  (selling_price : ℕ)
  (profit_per_metre : ℕ)
  (h1 : total_metres = 30)
  (h2 : selling_price = 4500)
  (h3 : profit_per_metre = 10) :
  (selling_price - total_metres * profit_per_metre) / total_metres = 140 :=
sorry

end cloth_cost_price_l1566_156617


namespace lemonade_stand_problem_l1566_156621

/-- Represents the lemonade stand problem --/
theorem lemonade_stand_problem 
  (glasses_per_gallon : ℕ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ)
  (net_profit : ℚ)
  (h1 : glasses_per_gallon = 16)
  (h2 : gallons_made = 2)
  (h3 : price_per_glass = 1)
  (h4 : glasses_drunk = 5)
  (h5 : glasses_unsold = 6)
  (h6 : net_profit = 14) :
  (gallons_made * glasses_per_gallon - glasses_drunk - glasses_unsold) * price_per_glass - net_profit = gallons_made * (7/2 : ℚ) :=
sorry

end lemonade_stand_problem_l1566_156621


namespace park_tree_increase_l1566_156659

/-- Represents the state of trees in the park -/
structure ParkState where
  maples : ℕ
  lindens : ℕ

/-- Calculates the total number of trees in the park -/
def total_trees (state : ParkState) : ℕ := state.maples + state.lindens

/-- Calculates the percentage of maples in the park -/
def maple_percentage (state : ParkState) : ℚ :=
  state.maples / (total_trees state)

/-- The initial state of the park -/
def initial_state : ParkState := sorry

/-- The state after planting lindens in spring -/
def spring_state : ParkState := sorry

/-- The final state after planting maples in autumn -/
def autumn_state : ParkState := sorry

theorem park_tree_increase :
  maple_percentage initial_state = 3/5 →
  maple_percentage spring_state = 1/5 →
  maple_percentage autumn_state = 3/5 →
  total_trees autumn_state = 6 * total_trees initial_state :=
sorry

end park_tree_increase_l1566_156659


namespace gcf_210_286_l1566_156686

theorem gcf_210_286 : Nat.gcd 210 286 = 2 := by
  sorry

end gcf_210_286_l1566_156686


namespace andrei_club_visits_l1566_156652

theorem andrei_club_visits :
  ∀ (d c : ℕ),
  15 * d + 11 * c = 115 →
  d + c = 9 :=
by
  sorry

end andrei_club_visits_l1566_156652


namespace wild_animal_population_estimation_l1566_156689

/-- Represents the data for a sample plot -/
structure PlotData where
  x : ℝ  -- plant coverage area
  y : ℝ  -- number of wild animals

/-- Represents the statistical data for the sample -/
structure SampleStats where
  n : ℕ              -- number of sample plots
  total_plots : ℕ    -- total number of plots in the area
  sum_x : ℝ          -- sum of x values
  sum_y : ℝ          -- sum of y values
  sum_x_squared : ℝ  -- sum of (x - x̄)²
  sum_y_squared : ℝ  -- sum of (y - ȳ)²
  sum_xy : ℝ         -- sum of (x - x̄)(y - ȳ)

/-- Theorem statement for the wild animal population estimation problem -/
theorem wild_animal_population_estimation
  (stats : SampleStats)
  (h1 : stats.n = 20)
  (h2 : stats.total_plots = 200)
  (h3 : stats.sum_x = 60)
  (h4 : stats.sum_y = 1200)
  (h5 : stats.sum_x_squared = 80)
  (h6 : stats.sum_y_squared = 9000)
  (h7 : stats.sum_xy = 800) :
  let estimated_population := (stats.sum_y / stats.n) * stats.total_plots
  let correlation_coefficient := stats.sum_xy / Real.sqrt (stats.sum_x_squared * stats.sum_y_squared)
  estimated_population = 12000 ∧ abs (correlation_coefficient - 0.94) < 0.01 := by
  sorry


end wild_animal_population_estimation_l1566_156689


namespace equation_solution_l1566_156669

theorem equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  let x : ℝ := (5 * m^2 + 18 * m * n + 18 * n^2) / (10 * m + 6 * n)
  (x + 2 * m)^2 - (x - 3 * n)^2 = 9 * (m + n)^2 := by
  sorry

end equation_solution_l1566_156669


namespace racing_car_A_time_l1566_156636

/-- The time taken by racing car A to complete the track -/
def time_A : ℕ := 7

/-- The time taken by racing car B to complete the track -/
def time_B : ℕ := 24

/-- The time after which both cars are side by side again -/
def side_by_side_time : ℕ := 168

/-- Theorem stating that the time taken by racing car A is correct -/
theorem racing_car_A_time :
  (time_A = 7) ∧ 
  (time_B = 24) ∧
  (side_by_side_time = 168) ∧
  (Nat.lcm time_A time_B = side_by_side_time) :=
sorry

end racing_car_A_time_l1566_156636


namespace simplify_and_evaluate_l1566_156643

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -2) (h2 : b = 3) :
  2 * (a^2 - a*b) - 3 * ((2/3) * a^2 - a*b - 1) = -3 := by
  sorry

end simplify_and_evaluate_l1566_156643


namespace c_alone_time_l1566_156612

-- Define the rates of work for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
axiom ab_rate : rA + rB = 1/3
axiom bc_rate : rB + rC = 1/6
axiom ac_rate : rA + rC = 1/4

-- Define the theorem
theorem c_alone_time : 1 / rC = 24 := by
  sorry

end c_alone_time_l1566_156612


namespace cube_sum_theorem_l1566_156683

theorem cube_sum_theorem (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 5) :
  a^3 + b^3 = 238 := by
sorry

end cube_sum_theorem_l1566_156683


namespace infinite_pairs_with_2020_diff_l1566_156666

/-- A positive integer is square-free if it is not divisible by any perfect square other than 1. -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → d * d ∣ n → d = 1

/-- The sequence of square-free positive integers in ascending order. -/
def SquareFreeSequence : ℕ → ℕ := sorry

/-- The property that all integers between two given numbers are not square-free. -/
def AllBetweenNotSquareFree (m n : ℕ) : Prop :=
  ∀ k : ℕ, m < k → k < n → ¬(IsSquareFree k)

/-- The main theorem stating that there are infinitely many pairs of consecutive
    square-free integers in the sequence with a difference of 2020. -/
theorem infinite_pairs_with_2020_diff :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧
    IsSquareFree (SquareFreeSequence n) ∧
    IsSquareFree (SquareFreeSequence (n + 1)) ∧
    SquareFreeSequence (n + 1) - SquareFreeSequence n = 2020 ∧
    AllBetweenNotSquareFree (SquareFreeSequence n) (SquareFreeSequence (n + 1)) :=
sorry

end infinite_pairs_with_2020_diff_l1566_156666


namespace snail_distance_is_20_l1566_156634

def snail_path : List ℤ := [0, 4, -3, 6]

def distance (a b : ℤ) : ℕ := (a - b).natAbs

def total_distance (path : List ℤ) : ℕ :=
  (path.zip path.tail).foldl (fun acc (a, b) => acc + distance a b) 0

theorem snail_distance_is_20 : total_distance snail_path = 20 := by
  sorry

end snail_distance_is_20_l1566_156634


namespace damaged_potatoes_calculation_l1566_156649

/-- Calculates the amount of damaged potatoes during transport -/
def damaged_potatoes (initial_amount : ℕ) (bag_size : ℕ) (price_per_bag : ℕ) (total_sales : ℕ) : ℕ :=
  initial_amount - (total_sales / price_per_bag * bag_size)

/-- Theorem stating the amount of damaged potatoes -/
theorem damaged_potatoes_calculation :
  damaged_potatoes 6500 50 72 9144 = 150 := by
  sorry

#eval damaged_potatoes 6500 50 72 9144

end damaged_potatoes_calculation_l1566_156649


namespace three_digit_base_nine_to_base_three_digit_count_sum_l1566_156675

theorem three_digit_base_nine_to_base_three_digit_count_sum :
  ∀ n : ℕ,
  (3^4 ≤ n ∧ n < 3^6) →
  (∃ e : ℕ, (3^(e-1) ≤ n ∧ n < 3^e) ∧ (e = 5 ∨ e = 6 ∨ e = 7)) ∧
  (5 + 6 + 7 = 18) :=
sorry

end three_digit_base_nine_to_base_three_digit_count_sum_l1566_156675


namespace certain_number_proof_l1566_156673

theorem certain_number_proof (p q x : ℝ) 
  (h1 : 3 / p = x)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.20833333333333334) :
  x = 8 := by
  sorry

end certain_number_proof_l1566_156673


namespace proportion_problem_l1566_156664

theorem proportion_problem (y : ℝ) : 0.75 / 0.9 = 5 / y → y = 6 := by
  sorry

end proportion_problem_l1566_156664


namespace regular_polygon_properties_l1566_156693

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides and interior angles of 162 degrees -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (exterior_angle interior_angle : ℝ),
  exterior_angle = 18 →
  n = (360 : ℝ) / exterior_angle →
  interior_angle = 180 - exterior_angle →
  n = 20 ∧ interior_angle = 162 :=
by
  sorry

end regular_polygon_properties_l1566_156693


namespace banana_arrangements_count_l1566_156670

/-- The number of unique arrangements of letters in "BANANA" -/
def banana_arrangements : ℕ := 
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of "BANANA" is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

#eval banana_arrangements

end banana_arrangements_count_l1566_156670


namespace car_speed_proof_l1566_156623

/-- Proves that a car's speed is 48 km/h if it takes 15 seconds longer to travel 1 km compared to 60 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v - 1 / 60) * 3600 = 15 ↔ v = 48 :=
by sorry

end car_speed_proof_l1566_156623


namespace intersection_solutions_l1566_156640

theorem intersection_solutions (α β : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ 2*x + 2*y - 1 - Real.sqrt 3 = 0 ∧
   (x = Real.sin α ∧ y = Real.sin (2*β) ∨ x = Real.sin β ∧ y = Real.cos (2*α))) →
  (∃ (n k : ℤ), α = (-1)^n * π/6 + π * (n : ℝ) ∧ β = π/3 + 2*π * (k : ℝ)) :=
by sorry

end intersection_solutions_l1566_156640


namespace root_product_bound_l1566_156667

-- Define the equations
def equation1 (x : ℝ) : Prop := Real.log x / Real.log 4 - (1/4)^x = 0
def equation2 (x : ℝ) : Prop := Real.log x / Real.log (1/4) - (1/4)^x = 0

-- State the theorem
theorem root_product_bound 
  (x₁ x₂ : ℝ) 
  (h1 : equation1 x₁) 
  (h2 : equation2 x₂) : 
  0 < x₁ * x₂ ∧ x₁ * x₂ < 1 := by
  sorry

end root_product_bound_l1566_156667


namespace initial_choir_size_l1566_156684

/-- The number of girls initially in the choir is equal to the sum of blonde-haired and black-haired girls. -/
theorem initial_choir_size (blonde_girls black_girls : ℕ) 
  (h1 : blonde_girls = 30) 
  (h2 : black_girls = 50) : 
  blonde_girls + black_girls = 80 := by
  sorry

end initial_choir_size_l1566_156684


namespace one_third_of_four_equals_two_l1566_156611

-- Define the country's multiplication operation
noncomputable def country_mul (a b : ℚ) : ℚ := sorry

-- Define the property that 1/8 of 4 equals 3 in this system
axiom country_property : country_mul (1/8) 4 = 3

-- Theorem statement
theorem one_third_of_four_equals_two : 
  country_mul (1/3) 4 = 2 := by sorry

end one_third_of_four_equals_two_l1566_156611


namespace tank_filling_time_l1566_156648

theorem tank_filling_time (fill_rate_A fill_rate_B : ℚ) : 
  fill_rate_A = 1 / 60 →
  15 * fill_rate_B + 15 * (fill_rate_A + fill_rate_B) = 1 →
  fill_rate_B = 1 / 40 :=
by sorry

end tank_filling_time_l1566_156648


namespace minimum_score_exists_l1566_156681

/-- Represents the scores of the four people who took the math test. -/
structure TestScores where
  marty : ℕ
  others : Fin 3 → ℕ

/-- The proposition that Marty's score is the minimum to conclude others scored below average. -/
def IsMinimumScore (scores : TestScores) : Prop :=
  scores.marty = 61 ∧
  (∀ i : Fin 3, scores.others i < 20) ∧
  (∀ s : TestScores, s.marty < 61 → 
    ∃ i : Fin 3, s.others i ≥ 20 ∨ (s.marty + (Finset.sum Finset.univ s.others)) / 4 ≠ 20)

/-- The theorem stating that there exists a score distribution satisfying the conditions. -/
theorem minimum_score_exists : ∃ scores : TestScores, IsMinimumScore scores ∧ 
  (scores.marty + (Finset.sum Finset.univ scores.others)) / 4 = 20 := by
  sorry

#check minimum_score_exists

end minimum_score_exists_l1566_156681


namespace rice_weight_scientific_notation_l1566_156668

theorem rice_weight_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 0.000035 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.5 ∧ n = -5 := by
  sorry

end rice_weight_scientific_notation_l1566_156668


namespace median_in_75_79_interval_l1566_156639

/-- Represents a score interval with its frequency --/
structure ScoreInterval :=
  (lower upper : ℕ)
  (frequency : ℕ)

/-- The list of score intervals for the test --/
def scoreDistribution : List ScoreInterval :=
  [⟨85, 89, 20⟩, ⟨80, 84, 18⟩, ⟨75, 79, 15⟩, ⟨70, 74, 12⟩,
   ⟨65, 69, 10⟩, ⟨60, 64, 8⟩, ⟨55, 59, 10⟩, ⟨50, 54, 7⟩]

/-- The total number of students --/
def totalStudents : ℕ := 100

/-- Function to calculate the cumulative frequency up to a given interval --/
def cumulativeFrequency (intervals : List ScoreInterval) (targetLower : ℕ) : ℕ :=
  (intervals.filter (fun i => i.lower ≥ targetLower)).foldl (fun acc i => acc + i.frequency) 0

/-- Theorem stating that the median is in the 75-79 interval --/
theorem median_in_75_79_interval :
  ∃ (median : ℕ), 75 ≤ median ∧ median ≤ 79 ∧
  cumulativeFrequency scoreDistribution 75 > totalStudents / 2 ∧
  cumulativeFrequency scoreDistribution 80 ≤ totalStudents / 2 :=
sorry

end median_in_75_79_interval_l1566_156639


namespace power_equation_solution_l1566_156614

theorem power_equation_solution : ∃! x : ℤ, (3 : ℝ) ^ 7 * (3 : ℝ) ^ x = 81 := by sorry

end power_equation_solution_l1566_156614


namespace geometric_sequence_common_ratio_l1566_156687

/-- A geometric sequence with a_2 = 8 and a_5 = 64 has a common ratio of 2 -/
theorem geometric_sequence_common_ratio : ∀ (a : ℕ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * a 1)  -- Definition of geometric sequence
  → a 2 = 8                         -- Given condition
  → a 5 = 64                        -- Given condition
  → a 1 = 2                         -- Common ratio q = a_1
  := by sorry

end geometric_sequence_common_ratio_l1566_156687


namespace original_price_proof_l1566_156633

-- Define the discount rates
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.05

-- Define the final sale price
def final_price : ℝ := 304

-- Theorem statement
theorem original_price_proof :
  ∃ (original_price : ℝ),
    original_price * (1 - first_discount) * (1 - second_discount) = final_price ∧
    original_price = 400 := by
  sorry

end original_price_proof_l1566_156633


namespace oranges_per_day_l1566_156658

/-- Proves that the number of sacks harvested per day is 4, given 56 sacks over 14 days -/
theorem oranges_per_day (total_sacks : ℕ) (total_days : ℕ) 
  (h1 : total_sacks = 56) (h2 : total_days = 14) : 
  total_sacks / total_days = 4 := by
  sorry

end oranges_per_day_l1566_156658


namespace seven_successes_probability_l1566_156655

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The number of successes -/
def k : ℕ := 7

/-- The probability of k successes in n Bernoulli trials with probability p -/
def bernoulli_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem stating the probability of 7 successes in 7 Bernoulli trials with p = 2/7 -/
theorem seven_successes_probability : 
  bernoulli_probability n k p = 128/823543 := by
  sorry

end seven_successes_probability_l1566_156655


namespace arithmetic_sequence_product_l1566_156657

/-- Given an arithmetic sequence {a_n}, if a_2^2 + 2a_2a_8 + a_6a_10 = 16, then a_4a_6 = 4 -/
theorem arithmetic_sequence_product (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 2^2 + 2 * a 2 * a 8 + a 6 * a 10 = 16 →
  a 4 * a 6 = 4 :=
by
  sorry

end arithmetic_sequence_product_l1566_156657


namespace arithmetic_sequence_sum_l1566_156698

theorem arithmetic_sequence_sum (d : ℝ) (h : d ≠ 0) :
  let a : ℕ → ℝ := fun n => (n - 1 : ℝ) * d
  ∃ m : ℕ, a m = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) ∧ m = 37 :=
by
  sorry

end arithmetic_sequence_sum_l1566_156698


namespace walk_time_proof_l1566_156696

/-- Ajay's walking speed in km/hour -/
def walking_speed : ℝ := 6

/-- The time taken to walk a certain distance in hours -/
def time_taken : ℝ := 11.666666666666666

/-- Theorem stating that the time taken to walk the distance is 11.666666666666666 hours -/
theorem walk_time_proof : time_taken = 11.666666666666666 := by
  sorry

end walk_time_proof_l1566_156696


namespace average_of_a_and_b_l1566_156690

theorem average_of_a_and_b (a b c : ℝ) : 
  ((b + c) / 2 = 180) → 
  (a - c = 200) → 
  ((a + b) / 2 = 280) :=
by
  sorry

end average_of_a_and_b_l1566_156690


namespace square_sum_given_sum_and_product_l1566_156641

theorem square_sum_given_sum_and_product (m n : ℝ) 
  (h1 : m + n = 10) (h2 : m * n = 24) : m^2 + n^2 = 52 := by
  sorry

end square_sum_given_sum_and_product_l1566_156641


namespace combined_platform_length_l1566_156653

/-- The combined length of two train platforms -/
theorem combined_platform_length
  (length_train_a : ℝ)
  (time_platform_a : ℝ)
  (time_pole_a : ℝ)
  (length_train_b : ℝ)
  (time_platform_b : ℝ)
  (time_pole_b : ℝ)
  (h1 : length_train_a = 500)
  (h2 : time_platform_a = 75)
  (h3 : time_pole_a = 25)
  (h4 : length_train_b = 400)
  (h5 : time_platform_b = 60)
  (h6 : time_pole_b = 20) :
  (length_train_a + (length_train_a / time_pole_a) * time_platform_a - length_train_a) +
  (length_train_b + (length_train_b / time_pole_b) * time_platform_b - length_train_b) = 1800 :=
by sorry

end combined_platform_length_l1566_156653


namespace qualification_rate_example_l1566_156622

/-- Calculates the qualification rate given the total number of boxes and the number of qualified boxes -/
def qualification_rate (total : ℕ) (qualified : ℕ) : ℚ :=
  (qualified : ℚ) / (total : ℚ) * 100

/-- Theorem stating that given 50 total boxes and 38 qualified boxes, the qualification rate is 76% -/
theorem qualification_rate_example : qualification_rate 50 38 = 76 := by
  sorry

end qualification_rate_example_l1566_156622


namespace blue_balls_count_l1566_156632

theorem blue_balls_count (total : ℕ) (removed : ℕ) (prob : ℚ) (initial : ℕ) : 
  total = 25 →
  removed = 5 →
  prob = 1/5 →
  (initial - removed : ℚ) / (total - removed : ℚ) = prob →
  initial = 9 := by sorry

end blue_balls_count_l1566_156632


namespace chromosome_variation_identification_l1566_156678

-- Define the structure for a genetic condition
structure GeneticCondition where
  name : String
  chromosomeAffected : Nat
  variationType : String

-- Define the statements
def statement1 : GeneticCondition := ⟨"cri-du-chat syndrome", 5, "partial deletion"⟩
def statement2 := "free combination of non-homologous chromosomes during meiosis"
def statement3 := "chromosomal exchange between synapsed homologous chromosomes"
def statement4 : GeneticCondition := ⟨"Down syndrome", 21, "extra chromosome"⟩

-- Define what constitutes a chromosome variation
def isChromosomeVariation (condition : GeneticCondition) : Prop :=
  condition.variationType = "partial deletion" ∨ condition.variationType = "extra chromosome"

-- Theorem to prove
theorem chromosome_variation_identification :
  (isChromosomeVariation statement1 ∧ isChromosomeVariation statement4) ∧
  (¬ isChromosomeVariation ⟨"", 0, statement2⟩ ∧ ¬ isChromosomeVariation ⟨"", 0, statement3⟩) := by
  sorry


end chromosome_variation_identification_l1566_156678


namespace problem_solution_l1566_156694

def X : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2017}

def S : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 ∈ X ∧ t.2.1 ∈ X ∧ t.2.2 ∈ X ∧
    ((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∨
     (t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∨
     (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.1 < t.2.2 ∧ t.2.2 < t.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1))}

theorem problem_solution (x y z w : ℕ) 
  (h1 : (x, y, z) ∈ S) (h2 : (z, w, x) ∈ S) :
  (y, z, w) ∈ S ∧ (x, y, w) ∈ S := by
  sorry

end problem_solution_l1566_156694


namespace quadratic_inequality_solution_l1566_156609

theorem quadratic_inequality_solution (x : ℝ) :
  -5 * x^2 + 7 * x + 2 > 0 ↔ 
  x > ((-7 - Real.sqrt 89) / -10) ∧ x < ((-7 + Real.sqrt 89) / -10) := by
sorry

end quadratic_inequality_solution_l1566_156609


namespace set_operations_and_subset_l1566_156638

def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  ((Bᶜ ∪ A) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x}) ∧
  (∀ a : ℝ, C a ⊆ B → (2 ≤ a ∧ a ≤ 8)) :=
sorry

end set_operations_and_subset_l1566_156638


namespace subgrids_cover_half_board_l1566_156630

/-- Represents a subgrid on the board -/
structure Subgrid where
  rows : ℕ
  cols : ℕ

/-- The board and its properties -/
structure Board where
  n : ℕ
  subgrids : List Subgrid

/-- Calculates the half-perimeter of a subgrid -/
def half_perimeter (s : Subgrid) : ℕ := s.rows + s.cols

/-- Checks if a list of subgrids covers the main diagonal -/
def covers_main_diagonal (b : Board) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ b.n → ∃ s ∈ b.subgrids, half_perimeter s ≥ b.n

/-- Calculates the number of squares covered by a list of subgrids -/
def squares_covered (b : Board) : ℕ :=
  sorry -- Implementation details omitted

/-- Main theorem -/
theorem subgrids_cover_half_board (b : Board) 
  (h_board_size : b.n * b.n = 11 * 60)
  (h_cover_diagonal : covers_main_diagonal b) :
  2 * (squares_covered b) ≥ b.n * b.n := by
  sorry

end subgrids_cover_half_board_l1566_156630
