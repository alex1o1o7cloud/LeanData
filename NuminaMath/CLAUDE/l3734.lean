import Mathlib

namespace NUMINAMATH_CALUDE_jordan_seven_miles_time_l3734_373454

/-- Jordan's running time for a given distance -/
def jordanTime (distance : ℝ) : ℝ := sorry

/-- Steve's running time for a given distance -/
def steveTime (distance : ℝ) : ℝ := sorry

/-- Theorem stating Jordan's time for 7 miles given the conditions -/
theorem jordan_seven_miles_time :
  (jordanTime 3 = 2 / 3 * steveTime 5) →
  (steveTime 5 = 40) →
  (∀ d₁ d₂ : ℝ, jordanTime d₁ / d₁ = jordanTime d₂ / d₂) →
  jordanTime 7 = 185 / 3 := by
  sorry

end NUMINAMATH_CALUDE_jordan_seven_miles_time_l3734_373454


namespace NUMINAMATH_CALUDE_min_trucks_required_l3734_373411

/-- Represents the weight capacity of a truck in tons -/
def truck_capacity : ℝ := 3

/-- Represents the total weight of all boxes in tons -/
def total_weight : ℝ := 10

/-- Represents the maximum weight of a single box in tons -/
def max_box_weight : ℝ := 1

/-- The minimum number of trucks required -/
def min_trucks : ℕ := 5

/-- Theorem stating that the minimum number of trucks required is 5 -/
theorem min_trucks_required :
  ∀ (box_weights : List ℝ),
    (box_weights.sum = total_weight) →
    (∀ w ∈ box_weights, w ≤ max_box_weight) →
    (∀ n : ℕ, n < min_trucks → n * truck_capacity < total_weight) →
    (min_trucks * truck_capacity ≥ total_weight) :=
by sorry

end NUMINAMATH_CALUDE_min_trucks_required_l3734_373411


namespace NUMINAMATH_CALUDE_second_class_average_l3734_373435

/-- Given two classes of students, this theorem proves that the average mark of the second class
    is 80, based on the given conditions. -/
theorem second_class_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 30 →
  n₂ = 50 →
  avg₁ = 40 →
  avg_total = 65 →
  let total_students : ℕ := n₁ + n₂
  let total_marks : ℚ := avg_total * total_students
  let first_class_marks : ℚ := avg₁ * n₁
  let second_class_marks : ℚ := total_marks - first_class_marks
  let avg₂ : ℚ := second_class_marks / n₂
  avg₂ = 80 := by sorry

end NUMINAMATH_CALUDE_second_class_average_l3734_373435


namespace NUMINAMATH_CALUDE_concert_problem_l3734_373448

/-- Represents the number of songs sung by each friend -/
structure SongCount where
  lucy : ℕ
  gina : ℕ
  zoe : ℕ
  sara : ℕ

/-- Calculates the total number of songs performed by the trios -/
def totalSongs (sc : SongCount) : ℚ :=
  (sc.lucy + sc.gina + sc.zoe + sc.sara) / 3

/-- Represents the conditions of the problem -/
def validSongCount (sc : SongCount) : Prop :=
  sc.sara = 9 ∧
  sc.lucy = 3 ∧
  sc.zoe = sc.sara ∧
  sc.gina > sc.lucy ∧
  sc.gina ≤ sc.sara ∧
  (sc.lucy + sc.gina) % 4 = 0

theorem concert_problem (sc : SongCount) (h : validSongCount sc) :
  totalSongs sc = 9 ∨ totalSongs sc = 10 := by
  sorry


end NUMINAMATH_CALUDE_concert_problem_l3734_373448


namespace NUMINAMATH_CALUDE_factorial_30_prime_factors_l3734_373489

theorem factorial_30_prime_factors : 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_30_prime_factors_l3734_373489


namespace NUMINAMATH_CALUDE_max_areas_circular_disk_l3734_373476

/-- 
Given a circular disk divided by 2n equally spaced radii and two secant lines 
that do not intersect at the same point on the circumference, the maximum number 
of non-overlapping areas into which the disk can be divided is 4n + 4.
-/
theorem max_areas_circular_disk (n : ℕ) : ℕ := by
  sorry

#check max_areas_circular_disk

end NUMINAMATH_CALUDE_max_areas_circular_disk_l3734_373476


namespace NUMINAMATH_CALUDE_nicole_bike_time_l3734_373405

/-- Given Nicole's biking information, calculate the time to ride 5 miles -/
theorem nicole_bike_time (distance_to_nathan : ℝ) (time_to_nathan : ℝ) (distance_to_patrick : ℝ)
  (h1 : distance_to_nathan = 2)
  (h2 : time_to_nathan = 8)
  (h3 : distance_to_patrick = 5) :
  distance_to_patrick / (distance_to_nathan / time_to_nathan) = 20 := by
  sorry

#check nicole_bike_time

end NUMINAMATH_CALUDE_nicole_bike_time_l3734_373405


namespace NUMINAMATH_CALUDE_initial_men_is_100_l3734_373473

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℝ
  completedLength : ℝ
  completedDays : ℝ
  extraMen : ℕ

/-- Calculates the initial number of men employed in the road project -/
def initialMenEmployed (project : RoadProject) : ℕ :=
  sorry

/-- Theorem stating that the initial number of men employed is 100 -/
theorem initial_men_is_100 (project : RoadProject) 
  (h1 : project.totalLength = 15)
  (h2 : project.totalDays = 300)
  (h3 : project.completedLength = 2.5)
  (h4 : project.completedDays = 100)
  (h5 : project.extraMen = 60) :
  initialMenEmployed project = 100 := by
  sorry

#check initial_men_is_100

end NUMINAMATH_CALUDE_initial_men_is_100_l3734_373473


namespace NUMINAMATH_CALUDE_square_difference_identity_l3734_373437

theorem square_difference_identity : (15 + 5)^2 - (15^2 + 5^2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l3734_373437


namespace NUMINAMATH_CALUDE_transformed_point_difference_l3734_373477

def rotate90CounterClockwise (x y xc yc : ℝ) : ℝ × ℝ :=
  (xc - (y - yc), yc + (x - xc))

def reflectAboutYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem transformed_point_difference (a b : ℝ) :
  let (x1, y1) := rotate90CounterClockwise a b 2 3
  let (x2, y2) := reflectAboutYEqualsX x1 y1
  (x2 = 4 ∧ y2 = 1) → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_transformed_point_difference_l3734_373477


namespace NUMINAMATH_CALUDE_three_W_five_l3734_373456

-- Define the operation W
def W (a b : ℤ) : ℤ := b + 7 * a - a ^ 2

-- Theorem to prove
theorem three_W_five : W 3 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_three_W_five_l3734_373456


namespace NUMINAMATH_CALUDE_rice_distribution_l3734_373474

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 35 / 2 →
  num_containers = 4 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce) / num_containers = 70 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_l3734_373474


namespace NUMINAMATH_CALUDE_right_triangle_geometric_sequence_ratio_l3734_373480

theorem right_triangle_geometric_sequence_ratio :
  ∀ (a b c : ℝ),
    a > 0 →
    b > 0 →
    c > 0 →
    a < b →
    b < c →
    a^2 + b^2 = c^2 →
    (∃ r : ℝ, r > 1 ∧ b = a * r ∧ c = a * r^2) →
    c / a = (1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_geometric_sequence_ratio_l3734_373480


namespace NUMINAMATH_CALUDE_small_cuboid_width_is_four_l3734_373450

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem small_cuboid_width_is_four
  (large : CuboidDimensions)
  (small_length : ℝ)
  (small_height : ℝ)
  (num_small_cuboids : ℕ)
  (h1 : large.length = 16)
  (h2 : large.width = 10)
  (h3 : large.height = 12)
  (h4 : small_length = 5)
  (h5 : small_height = 3)
  (h6 : num_small_cuboids = 32)
  (h7 : ∃ (small_width : ℝ),
    cuboidVolume large = num_small_cuboids * cuboidVolume
      { length := small_length
        width := small_width
        height := small_height }) :
  ∃ (small_width : ℝ), small_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_cuboid_width_is_four_l3734_373450


namespace NUMINAMATH_CALUDE_janet_earnings_per_hour_l3734_373467

-- Define the payment rates for each type of post
def text_post_rate : ℚ := 0.25
def image_post_rate : ℚ := 0.30
def video_post_rate : ℚ := 0.40

-- Define the number of posts checked in an hour
def text_posts_per_hour : ℕ := 130
def image_posts_per_hour : ℕ := 90
def video_posts_per_hour : ℕ := 30

-- Define the USD to EUR exchange rate
def usd_to_eur_rate : ℚ := 0.85

-- Calculate the earnings per hour in EUR
def earnings_per_hour_eur : ℚ :=
  (text_post_rate * text_posts_per_hour +
   image_post_rate * image_posts_per_hour +
   video_post_rate * video_posts_per_hour) * usd_to_eur_rate

-- Theorem to prove
theorem janet_earnings_per_hour :
  earnings_per_hour_eur = 60.775 := by sorry

end NUMINAMATH_CALUDE_janet_earnings_per_hour_l3734_373467


namespace NUMINAMATH_CALUDE_a_share_of_profit_l3734_373449

/-- Calculate A's share of the profit in a partnership business -/
theorem a_share_of_profit (a_investment b_investment c_investment total_profit : ℕ) :
  a_investment = 6300 →
  b_investment = 4200 →
  c_investment = 10500 →
  total_profit = 12700 →
  (a_investment * total_profit) / (a_investment + b_investment + c_investment) = 3810 :=
by sorry

end NUMINAMATH_CALUDE_a_share_of_profit_l3734_373449


namespace NUMINAMATH_CALUDE_probability_less_equal_nine_l3734_373445

def card_set : Finset ℕ := {1, 3, 4, 6, 7, 9}

theorem probability_less_equal_nine : 
  (card_set.filter (λ x => x ≤ 9)).card / card_set.card = 1 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_equal_nine_l3734_373445


namespace NUMINAMATH_CALUDE_min_value_theorem_l3734_373481

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + a*b + a*c + b*c = 6 + 2 * Real.sqrt 5) :
  3*a + b + 2*c ≥ 2 * Real.sqrt 10 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3734_373481


namespace NUMINAMATH_CALUDE_frank_cans_total_l3734_373470

/-- The number of cans Frank picked up on Saturday and Sunday combined -/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem stating that Frank picked up 40 cans in total -/
theorem frank_cans_total : total_cans 5 3 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_frank_cans_total_l3734_373470


namespace NUMINAMATH_CALUDE_reservoir_ratio_l3734_373443

theorem reservoir_ratio : 
  ∀ (total_capacity normal_level end_month_amount : ℝ),
  end_month_amount = 6 →
  end_month_amount = 0.6 * total_capacity →
  normal_level = total_capacity - 5 →
  end_month_amount / normal_level = 1.2 := by
sorry

end NUMINAMATH_CALUDE_reservoir_ratio_l3734_373443


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_two_l3734_373459

/-- A polynomial function of degree 5 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

/-- Theorem stating that f(-2) = -2 given the conditions -/
theorem f_neg_two_eq_neg_two (a b c : ℝ) :
  (f a b c 5 + f a b c (-5) = 6) →
  (f a b c 2 = 8) →
  f a b c (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_two_l3734_373459


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_9_18_36_l3734_373484

theorem gcf_lcm_sum_9_18_36 : 
  let A := Nat.gcd 9 (Nat.gcd 18 36)
  let B := Nat.lcm 9 (Nat.lcm 18 36)
  A + B = 45 := by sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_9_18_36_l3734_373484


namespace NUMINAMATH_CALUDE_farm_tax_percentage_l3734_373420

theorem farm_tax_percentage (total_tax collection_tax : ℝ) : 
  total_tax > 0 → 
  collection_tax > 0 → 
  collection_tax ≤ total_tax → 
  (collection_tax / total_tax) * 100 = 12.5 → 
  total_tax = 3840 ∧ collection_tax = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_tax_percentage_l3734_373420


namespace NUMINAMATH_CALUDE_theresas_work_hours_l3734_373496

theorem theresas_work_hours : ∀ (final_week_hours : ℕ),
  final_week_hours ≥ 10 →
  (7 + 10 + 8 + 11 + 9 + 7 + final_week_hours) / 7 = 9 →
  final_week_hours = 11 := by
sorry

end NUMINAMATH_CALUDE_theresas_work_hours_l3734_373496


namespace NUMINAMATH_CALUDE_hunting_ratio_l3734_373413

theorem hunting_ratio : 
  ∀ (sam rob mark peter total : ℕ) (mark_fraction : ℚ),
    sam = 6 →
    rob = sam / 2 →
    mark = mark_fraction * (sam + rob) →
    peter = 3 * mark →
    sam + rob + mark + peter = 21 →
    mark_fraction = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_hunting_ratio_l3734_373413


namespace NUMINAMATH_CALUDE_sum_product_inequality_l3734_373491

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + a * c + b * c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l3734_373491


namespace NUMINAMATH_CALUDE_overall_gain_loss_percent_zero_l3734_373465

def article_A_cost : ℝ := 600
def article_B_cost : ℝ := 700
def article_C_cost : ℝ := 800
def article_A_sell : ℝ := 450
def article_B_sell : ℝ := 750
def article_C_sell : ℝ := 900

def total_cost : ℝ := article_A_cost + article_B_cost + article_C_cost
def total_sell : ℝ := article_A_sell + article_B_sell + article_C_sell

theorem overall_gain_loss_percent_zero :
  (total_sell - total_cost) / total_cost * 100 = 0 := by sorry

end NUMINAMATH_CALUDE_overall_gain_loss_percent_zero_l3734_373465


namespace NUMINAMATH_CALUDE_root_implies_a_range_l3734_373439

def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x + 4

theorem root_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ f a x = 0) → a ∈ Set.Icc (-2) 1 := by
sorry

end NUMINAMATH_CALUDE_root_implies_a_range_l3734_373439


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_14_l3734_373422

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digit_sum_14 :
  ∀ n : ℕ, 
    is_valid_number n → 
    digit_sum n = 14 → 
    n ≤ 3222233 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_14_l3734_373422


namespace NUMINAMATH_CALUDE_practice_coincidence_l3734_373404

def trumpet_interval : ℕ := 11
def flute_interval : ℕ := 3

theorem practice_coincidence : Nat.lcm trumpet_interval flute_interval = 33 := by
  sorry

end NUMINAMATH_CALUDE_practice_coincidence_l3734_373404


namespace NUMINAMATH_CALUDE_tree_leaves_theorem_l3734_373440

/-- Calculates the number of leaves remaining on a tree after three weeks of shedding --/
def leaves_remaining (initial_leaves : ℕ) : ℕ :=
  let first_week_remaining := initial_leaves - (2 * initial_leaves / 5)
  let second_week_shed := (40 * first_week_remaining) / 100
  let second_week_remaining := first_week_remaining - second_week_shed
  let third_week_shed := (3 * second_week_shed) / 4
  second_week_remaining - third_week_shed

/-- Theorem stating that a tree with 1000 initial leaves will have 180 leaves remaining after three weeks of shedding --/
theorem tree_leaves_theorem : leaves_remaining 1000 = 180 := by
  sorry

end NUMINAMATH_CALUDE_tree_leaves_theorem_l3734_373440


namespace NUMINAMATH_CALUDE_parabola_directrix_l3734_373490

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y = (x^2 - 4*x + 4) / 8

/-- The directrix equation -/
def directrix_eq (y : ℝ) : Prop := y = -1/4

/-- Theorem: The directrix of the given parabola is y = -1/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → ∃ y_d : ℝ, directrix_eq y_d ∧ 
  (∀ x' y' : ℝ, parabola_eq x' y' → 
    (x' - x)^2 + (y' - y)^2 = (y' - y_d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3734_373490


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l3734_373468

theorem point_in_third_quadrant (m : ℝ) : 
  let P : ℝ × ℝ := (-m^2 - 1, -1)
  P.1 < 0 ∧ P.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l3734_373468


namespace NUMINAMATH_CALUDE_min_cost_is_58984_l3734_373471

/-- Represents a travel agency with its pricing structure -/
structure TravelAgency where
  name : String
  young_age_limit : Nat
  young_price : Nat
  adult_price : Nat
  discount_or_commission : Float
  is_discount : Bool

/-- Represents a family member -/
structure FamilyMember where
  age : Nat

/-- Calculates the total cost for a family's vacation with a given travel agency -/
def calculate_total_cost (agency : TravelAgency) (family : List FamilyMember) : Float :=
  sorry

/-- The Dorokhov family -/
def dorokhov_family : List FamilyMember :=
  [⟨35⟩, ⟨35⟩, ⟨5⟩]  -- Assuming parents are 35 years old

/-- Globus travel agency -/
def globus : TravelAgency :=
  ⟨"Globus", 5, 11200, 25400, 0.02, true⟩

/-- Around the World travel agency -/
def around_the_world : TravelAgency :=
  ⟨"Around the World", 6, 11400, 23500, 0.01, false⟩

/-- Theorem: The minimum cost for the Dorokhov family's vacation is 58984 rubles -/
theorem min_cost_is_58984 :
  min (calculate_total_cost globus dorokhov_family)
      (calculate_total_cost around_the_world dorokhov_family) = 58984 :=
  sorry

end NUMINAMATH_CALUDE_min_cost_is_58984_l3734_373471


namespace NUMINAMATH_CALUDE_seeds_per_small_garden_l3734_373472

/-- Proves that given the initial number of seeds, seeds planted in the big garden,
    and the number of small gardens, the number of seeds in each small garden is correct. -/
theorem seeds_per_small_garden 
  (total_seeds : ℕ) 
  (big_garden_seeds : ℕ) 
  (small_gardens : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : big_garden_seeds = 35)
  (h3 : small_gardens = 7)
  : (total_seeds - big_garden_seeds) / small_gardens = 3 := by
  sorry

end NUMINAMATH_CALUDE_seeds_per_small_garden_l3734_373472


namespace NUMINAMATH_CALUDE_magical_stack_with_79_fixed_l3734_373402

/-- A stack of cards is magical if it satisfies certain conditions -/
def magical_stack (n : ℕ) : Prop :=
  ∃ (card_position : ℕ → ℕ),
    (∀ i, i ≤ 2*n → card_position i ≤ 2*n) ∧
    (∃ i ≤ n, card_position i = i) ∧
    (∃ i > n, i ≤ 2*n ∧ card_position i = i) ∧
    (∀ i ≤ 2*n, i % 2 = 1 → card_position i ≤ n) ∧
    (∀ i ≤ 2*n, i % 2 = 0 → card_position i > n)

theorem magical_stack_with_79_fixed (n : ℕ) :
  magical_stack n ∧ n ≥ 79 ∧ (∃ card_position : ℕ → ℕ, card_position 79 = 79) →
  2 * n = 236 :=
sorry

end NUMINAMATH_CALUDE_magical_stack_with_79_fixed_l3734_373402


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l3734_373438

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * x^2 - 32 = 0
def equation2 (x : ℝ) : Prop := (x + 4)^3 + 64 = 0

-- Theorem for the first equation
theorem equation1_solutions :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = 4 ∧ x₂ = -4 :=
sorry

-- Theorem for the second equation
theorem equation2_solution :
  ∃ x : ℝ, equation2 x ∧ x = -8 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l3734_373438


namespace NUMINAMATH_CALUDE_cara_seating_arrangement_l3734_373401

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem cara_seating_arrangement :
  choose 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangement_l3734_373401


namespace NUMINAMATH_CALUDE_all_boys_are_brothers_l3734_373427

/-- A type representing the group of boys -/
def Boys := Fin 7

/-- A relation indicating whether two boys are brothers -/
def is_brother (a b : Boys) : Prop := sorry

/-- Axiom: Each boy has at least 3 brothers among the others -/
axiom at_least_three_brothers (b : Boys) : 
  ∃ (s : Finset Boys), s.card ≥ 3 ∧ ∀ x ∈ s, x ≠ b ∧ is_brother x b

/-- Theorem: All seven boys are brothers -/
theorem all_boys_are_brothers : ∀ (a b : Boys), is_brother a b :=
sorry

end NUMINAMATH_CALUDE_all_boys_are_brothers_l3734_373427


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_multiple_condition_l3734_373414

theorem greatest_three_digit_number_multiple_condition : ∃ n : ℕ,
  (n ≤ 999) ∧ 
  (n ≥ 100) ∧
  (∃ k : ℕ, n = 9 * k + 2) ∧
  (∃ m : ℕ, n = 7 * m + 4) ∧
  (∀ x : ℕ, x ≤ 999 ∧ x ≥ 100 ∧ (∃ k : ℕ, x = 9 * k + 2) ∧ (∃ m : ℕ, x = 7 * m + 4) → x ≤ n) ∧
  n = 956 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_multiple_condition_l3734_373414


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3734_373415

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Black : Card
| White : Card
| Blue : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person B gets the red card"
def event_B_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Define mutual exclusivity
def mutually_exclusive (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, ¬(e1 d ∧ e2 d)

-- Define complementary events
def complementary (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, e1 d ↔ ¬(e2 d)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutually_exclusive event_A_red event_B_red ∧
  ¬(complementary event_A_red event_B_red) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3734_373415


namespace NUMINAMATH_CALUDE_teachers_in_middle_probability_l3734_373463

def num_students : ℕ := 3
def num_teachers : ℕ := 2
def num_parents : ℕ := 3
def total_people : ℕ := num_students + num_teachers + num_parents

def probability_teachers_in_middle : ℚ :=
  (Nat.factorial (total_people - num_teachers)) / (Nat.factorial total_people)

theorem teachers_in_middle_probability :
  probability_teachers_in_middle = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_teachers_in_middle_probability_l3734_373463


namespace NUMINAMATH_CALUDE_right_triangle_identification_l3734_373499

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_identification :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 5 12 13 ∧
  is_right_triangle 6 8 10 ∧
  ¬ is_right_triangle 4 6 8 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l3734_373499


namespace NUMINAMATH_CALUDE_triangle_midpoint_sum_l3734_373403

theorem triangle_midpoint_sum (a b c : ℝ) : 
  a + b + c = 15 → 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_midpoint_sum_l3734_373403


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3734_373444

theorem sin_2theta_value (θ : Real) (h : Real.cos θ + Real.sin θ = 3/2) : 
  Real.sin (2 * θ) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3734_373444


namespace NUMINAMATH_CALUDE_wine_bottle_cost_l3734_373464

/-- The cost of a bottle of wine with a cork, given the price of the cork and the price difference between a bottle without a cork and the cork itself. -/
theorem wine_bottle_cost (cork_price : ℝ) (price_difference : ℝ) : 
  cork_price = 0.05 →
  price_difference = 2.00 →
  cork_price + (cork_price + price_difference) = 2.10 :=
by sorry

end NUMINAMATH_CALUDE_wine_bottle_cost_l3734_373464


namespace NUMINAMATH_CALUDE_division_of_monomials_l3734_373447

theorem division_of_monomials (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  10 * a^3 * b^2 / (-5 * a^2 * b) = -2 * a * b :=
by sorry

end NUMINAMATH_CALUDE_division_of_monomials_l3734_373447


namespace NUMINAMATH_CALUDE_work_comparison_l3734_373424

/-- Represents the amount of work that can be done by a group of people in a given number of days -/
structure WorkCapacity where
  people : ℕ
  days : ℕ
  work : ℝ

/-- The work capacity is directly proportional to the number of people and days -/
axiom work_proportional {w1 w2 : WorkCapacity} : 
  w1.work / w2.work = (w1.people * w1.days : ℝ) / (w2.people * w2.days)

theorem work_comparison (w1 w2 : WorkCapacity) 
  (h1 : w1.people = 3 ∧ w1.days = 3)
  (h2 : w2.people = 8 ∧ w2.days = 3)
  (h3 : w2.work = 8 * w1.work) :
  w1.work = 3 * w1.work := by
  sorry

end NUMINAMATH_CALUDE_work_comparison_l3734_373424


namespace NUMINAMATH_CALUDE_topsoil_cost_l3734_373478

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 6

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards_of_topsoil : ℝ := 5

/-- The theorem stating the cost of the given amount of topsoil -/
theorem topsoil_cost : 
  cubic_yards_of_topsoil * cubic_feet_per_cubic_yard * topsoil_cost_per_cubic_foot = 810 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l3734_373478


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3734_373419

theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3734_373419


namespace NUMINAMATH_CALUDE_kendy_bank_transactions_l3734_373406

theorem kendy_bank_transactions (X : ℝ) : 
  X - 60 - 30 - 0.25 * (X - 60 - 30) - 10 = 100 → X = 236.67 :=
by sorry

end NUMINAMATH_CALUDE_kendy_bank_transactions_l3734_373406


namespace NUMINAMATH_CALUDE_b_value_l3734_373466

def consecutive_odd_numbers (a b c d e : ℤ) : Prop :=
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2

theorem b_value (a b c d e : ℤ) 
  (h1 : consecutive_odd_numbers a b c d e)
  (h2 : a + c = 146)
  (h3 : e = 79) : 
  b = 73 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l3734_373466


namespace NUMINAMATH_CALUDE_perpendicular_tangents_imply_a_value_l3734_373460

/-- Given two curves C₁ and C₂, where C₁ is y = ax³ - x² + 2x and C₂ is y = e^x,
    if their tangent lines are perpendicular at x = 1, then a = -1/(3e) -/
theorem perpendicular_tangents_imply_a_value (a : ℝ) :
  let C₁ : ℝ → ℝ := λ x ↦ a * x^3 - x^2 + 2*x
  let C₂ : ℝ → ℝ := λ x ↦ Real.exp x
  let tangent_C₁ : ℝ := 3*a - 2 + 2  -- Derivative of C₁ at x = 1
  let tangent_C₂ : ℝ := Real.exp 1   -- Derivative of C₂ at x = 1
  (tangent_C₁ * tangent_C₂ = -1) →   -- Condition for perpendicular tangents
  a = -1 / (3 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_imply_a_value_l3734_373460


namespace NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l3734_373479

/-- Given a > 0 and f(x) = x^3 - ax is monotonically increasing on [1, +∞),
    prove that the range of values for a is (0, 3]. -/
theorem range_of_a_for_monotonic_f (a : ℝ) (f : ℝ → ℝ) :
  a > 0 →
  (∀ x, f x = x^3 - a*x) →
  (∀ x y, 1 ≤ x → x < y → f x < f y) →
  ∃ S, S = Set.Ioo 0 3 ∧ a ∈ S :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l3734_373479


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_sum_of_squares_l3734_373498

theorem two_numbers_sum_and_sum_of_squares (a b : ℝ) :
  (∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ (x : ℝ) + y = a ∧ (x : ℝ)^2 + y^2 = b) ↔
  (∃ (k : ℕ), 2*b - a^2 = (k : ℝ)^2 ∧ k > 0) :=
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_sum_of_squares_l3734_373498


namespace NUMINAMATH_CALUDE_tinsel_count_l3734_373497

/-- The number of pieces of tinsel in each box of Christmas decorations. -/
def tinsel_per_box : ℕ := 4

/-- The number of Christmas trees in each box. -/
def trees_per_box : ℕ := 1

/-- The number of snow globes in each box. -/
def snow_globes_per_box : ℕ := 5

/-- The total number of boxes distributed. -/
def total_boxes : ℕ := 12

/-- The total number of decorations handed out. -/
def total_decorations : ℕ := 120

/-- Theorem stating that the number of pieces of tinsel in each box is 4. -/
theorem tinsel_count : 
  total_boxes * (tinsel_per_box + trees_per_box + snow_globes_per_box) = total_decorations :=
by sorry

end NUMINAMATH_CALUDE_tinsel_count_l3734_373497


namespace NUMINAMATH_CALUDE_gcd_of_1054_and_986_l3734_373416

theorem gcd_of_1054_and_986 : Nat.gcd 1054 986 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_1054_and_986_l3734_373416


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l3734_373488

theorem simplify_sqrt_fraction : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l3734_373488


namespace NUMINAMATH_CALUDE_nested_function_evaluation_l3734_373425

def a (k : ℕ) : ℕ := (k + 1)^2

theorem nested_function_evaluation :
  let k : ℕ := 1
  a (a (a (a k))) = 458329 := by
  sorry

end NUMINAMATH_CALUDE_nested_function_evaluation_l3734_373425


namespace NUMINAMATH_CALUDE_units_digit_factorial_product_15_l3734_373492

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def productFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.foldl (·*·) 1

theorem units_digit_factorial_product_15 :
  unitsDigit (productFactorials 15) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_product_15_l3734_373492


namespace NUMINAMATH_CALUDE_certain_number_proof_l3734_373452

theorem certain_number_proof (x : ℚ) (n : ℚ) : 
  x = 6 → 9 - (4/x) = n + (8/x) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3734_373452


namespace NUMINAMATH_CALUDE_age_of_b_l3734_373446

/-- Given three people a, b, and c, prove that if their average age is 25 years
    and the average age of a and c is 29 years, then the age of b is 17 years. -/
theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 25 → (a + c) / 2 = 29 → b = 17 := by
  sorry

end NUMINAMATH_CALUDE_age_of_b_l3734_373446


namespace NUMINAMATH_CALUDE_skipping_competition_probability_l3734_373412

theorem skipping_competition_probability :
  let total_boys : ℕ := 4
  let total_girls : ℕ := 6
  let selected_boys : ℕ := 2
  let selected_girls : ℕ := 2
  let total_selections : ℕ := (Nat.choose total_boys selected_boys) * (Nat.choose total_girls selected_girls)
  let selections_without_A_and_B : ℕ := (Nat.choose (total_boys - 1) selected_boys) * (Nat.choose (total_girls - 1) selected_girls)
  (total_selections - selections_without_A_and_B) / total_selections = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_skipping_competition_probability_l3734_373412


namespace NUMINAMATH_CALUDE_college_choices_theorem_l3734_373431

/-- The number of colleges --/
def n : ℕ := 6

/-- The number of colleges to be chosen --/
def k : ℕ := 3

/-- The number of colleges with scheduling conflict --/
def conflict : ℕ := 2

/-- Function to calculate the number of ways to choose colleges --/
def chooseColleges (n k conflict : ℕ) : ℕ :=
  Nat.choose (n - conflict) k + conflict * Nat.choose (n - conflict) (k - 1)

/-- Theorem stating that the number of ways to choose colleges is 16 --/
theorem college_choices_theorem :
  chooseColleges n k conflict = 16 := by sorry

end NUMINAMATH_CALUDE_college_choices_theorem_l3734_373431


namespace NUMINAMATH_CALUDE_total_octopus_legs_l3734_373493

/-- The number of octopuses Sawyer saw -/
def num_octopuses : ℕ := 5

/-- The number of legs each octopus has -/
def legs_per_octopus : ℕ := 8

/-- The total number of octopus legs -/
def total_legs : ℕ := num_octopuses * legs_per_octopus

theorem total_octopus_legs : total_legs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_octopus_legs_l3734_373493


namespace NUMINAMATH_CALUDE_min_value_theorem_l3734_373453

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 6 ∧ ∀ (x : ℝ), x = 3/(a-1) + 2/(b-1) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3734_373453


namespace NUMINAMATH_CALUDE_expression_equals_three_l3734_373436

-- Define the expression
def expression : ℚ := -25 + 7 * ((8 / 4) ^ 2)

-- Theorem statement
theorem expression_equals_three : expression = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l3734_373436


namespace NUMINAMATH_CALUDE_percentage_without_full_time_jobs_survey_result_l3734_373441

theorem percentage_without_full_time_jobs 
  (mother_ratio : Real) 
  (father_ratio : Real) 
  (women_ratio : Real) : Real :=
  let total_parents := 100
  let women_count := women_ratio * total_parents
  let men_count := total_parents - women_count
  let employed_women := mother_ratio * women_count
  let employed_men := father_ratio * men_count
  let total_employed := employed_women + employed_men
  let unemployed := total_parents - total_employed
  unemployed / total_parents * 100

theorem survey_result : 
  percentage_without_full_time_jobs (5/6) (3/4) 0.6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_without_full_time_jobs_survey_result_l3734_373441


namespace NUMINAMATH_CALUDE_germs_per_dish_l3734_373429

theorem germs_per_dish :
  let total_germs : ℝ := 5.4 * 10^6
  let total_dishes : ℝ := 10800
  let germs_per_dish : ℝ := total_germs / total_dishes
  germs_per_dish = 500 :=
by sorry

end NUMINAMATH_CALUDE_germs_per_dish_l3734_373429


namespace NUMINAMATH_CALUDE_oranges_taken_away_l3734_373400

/-- Represents the number of fruits in Tina's bag -/
structure FruitBag where
  apples : Nat
  oranges : Nat
  tangerines : Nat

/-- Represents the number of fruits Tina took away -/
structure FruitsTakenAway where
  oranges : Nat
  tangerines : Nat

def initial_bag : FruitBag := { apples := 9, oranges := 5, tangerines := 17 }

def fruits_taken : FruitsTakenAway := { oranges := 2, tangerines := 10 }

theorem oranges_taken_away (bag : FruitBag) (taken : FruitsTakenAway) : 
  taken.oranges = 2 ↔ 
    (bag.tangerines - taken.tangerines = (bag.oranges - taken.oranges) + 4) ∧
    (taken.tangerines = 10) ∧
    (bag = initial_bag) := by
  sorry

end NUMINAMATH_CALUDE_oranges_taken_away_l3734_373400


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3734_373486

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

/-- The point of tangency -/
def P : ℝ × ℝ := (-1, 3)

/-- The slope of the tangent line at point P -/
def m : ℝ := f' P.1

/-- The equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := m * (x - P.1) + P.2

theorem tangent_line_equation :
  ∀ x : ℝ, tangent_line x = -4 * x - 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3734_373486


namespace NUMINAMATH_CALUDE_sequence_term_number_l3734_373409

theorem sequence_term_number : 
  let a : ℕ → ℝ := fun n => Real.sqrt (2 * n - 1)
  ∃ n : ℕ, n = 23 ∧ a n = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_number_l3734_373409


namespace NUMINAMATH_CALUDE_square_park_fencing_cost_l3734_373487

/-- The cost of fencing one side of a square park -/
def cost_per_side : ℕ := 43

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- The total cost of fencing a square park -/
def total_cost : ℕ := cost_per_side * num_sides

/-- Theorem: The total cost of fencing a square park is $172 -/
theorem square_park_fencing_cost :
  total_cost = 172 := by
  sorry

end NUMINAMATH_CALUDE_square_park_fencing_cost_l3734_373487


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3734_373430

theorem units_digit_of_product (n : ℕ) : 
  (2^2101 * 5^2102 * 11^2103) % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3734_373430


namespace NUMINAMATH_CALUDE_circle_translation_l3734_373421

/-- Given a circle equation, prove its center, radius, and translated form -/
theorem circle_translation (x y : ℝ) :
  let original_eq := x^2 + y^2 - 4*x + 6*y - 68 = 0
  let center := (2, -3)
  let radius := 9
  let X := x - 2
  let Y := y + 3
  let translated_eq := X^2 + Y^2 = 81
  original_eq → (
    (x - center.1)^2 + (y - center.2)^2 = radius^2 ∧
    translated_eq
  ) := by sorry

end NUMINAMATH_CALUDE_circle_translation_l3734_373421


namespace NUMINAMATH_CALUDE_hyperbola_right_directrix_l3734_373485

/-- Given a parabola and a hyperbola with a shared focus, this theorem proves 
    the equation of the right directrix of the hyperbola. -/
theorem hyperbola_right_directrix 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_focus : ∀ x y : ℝ, y^2 = 8*x → x^2/a^2 - y^2/3 = 1 → x = 2 ∧ y = 0) :
  ∃ x : ℝ, x = 1/2 ∧ 
    ∀ y : ℝ, (∃ t : ℝ, t^2/a^2 - y^2/3 = 1 ∧ t > x) → 
      x = a^2 / (2 * (a^2 + 3)^(1/2)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_right_directrix_l3734_373485


namespace NUMINAMATH_CALUDE_complex_magnitude_l3734_373428

theorem complex_magnitude (w : ℂ) (h : w^2 + 2*w = 11 - 16*I) : 
  Complex.abs w = 17 ∨ Complex.abs w = Real.sqrt 89 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3734_373428


namespace NUMINAMATH_CALUDE_min_sum_squares_l3734_373461

theorem min_sum_squares (x y z : ℝ) (h : 2 * x + 3 * y + 3 * z = 1) :
  (∀ a b c : ℝ, 2 * a + 3 * b + 3 * c = 1 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) →
  x^2 + y^2 + z^2 = 1 / 22 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3734_373461


namespace NUMINAMATH_CALUDE_class_test_probability_l3734_373457

theorem class_test_probability (p_first p_second p_neither : ℝ) 
  (h1 : p_first = 0.63)
  (h2 : p_second = 0.49)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.32 := by
    sorry

end NUMINAMATH_CALUDE_class_test_probability_l3734_373457


namespace NUMINAMATH_CALUDE_fibonacci_eight_sum_not_equal_single_l3734_373418

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_eight_sum_not_equal_single (k : ℕ) : 
  ¬∃ m : ℕ, 
    fibonacci k + fibonacci (k + 1) + fibonacci (k + 2) + fibonacci (k + 3) + 
    fibonacci (k + 4) + fibonacci (k + 5) + fibonacci (k + 6) + fibonacci (k + 7) = 
    fibonacci m := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_eight_sum_not_equal_single_l3734_373418


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_l3734_373495

/-- Represents a triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  is_triangle : a + b > c ∧ b + c > a ∧ c + a > b

/-- A triangle satisfying the given conditions -/
def SpecialTriangle (t : IntTriangle) : Prop :=
  (t.a = 4 * t.b ∨ t.b = 4 * t.c ∨ t.c = 4 * t.a) ∧ (t.a = 20 ∨ t.b = 20 ∨ t.c = 20)

/-- The perimeter of a triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Theorem stating the maximum perimeter of the special triangle -/
theorem max_perimeter_special_triangle :
  ∀ t : IntTriangle, SpecialTriangle t → perimeter t ≤ 50 :=
sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_l3734_373495


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3734_373417

/-- The area of a regular square inscribed in a circle with area 324π is 648 square units. -/
theorem inscribed_square_area (circle_area : ℝ) (h : circle_area = 324 * Real.pi) :
  let r : ℝ := Real.sqrt (circle_area / Real.pi)
  let square_side : ℝ := Real.sqrt 2 * r
  square_side ^ 2 = 648 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3734_373417


namespace NUMINAMATH_CALUDE_divisibility_condition_l3734_373462

theorem divisibility_condition (a b : ℕ+) : 
  (∃ k : ℕ, (a^2 * b + a + b : ℕ) = k * (a * b^2 + b + 7)) ↔ 
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3734_373462


namespace NUMINAMATH_CALUDE_stratified_sampling_group_b_l3734_373451

theorem stratified_sampling_group_b (total_cities : ℕ) (group_b_cities : ℕ) (sample_size : ℕ) :
  total_cities = 24 →
  group_b_cities = 12 →
  sample_size = 6 →
  (group_b_cities * sample_size) / total_cities = 3 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_b_l3734_373451


namespace NUMINAMATH_CALUDE_magician_tricks_conversion_l3734_373434

/-- Converts a base-9 number represented as a list of digits to its base-10 equivalent -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The given number of tricks in base 9 -/
def tricksBase9 : List Nat := [2, 3, 4, 5]

theorem magician_tricks_conversion :
  base9ToBase10 tricksBase9 = 3998 := by
  sorry

end NUMINAMATH_CALUDE_magician_tricks_conversion_l3734_373434


namespace NUMINAMATH_CALUDE_intersection_A_B_l3734_373408

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≥ 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3734_373408


namespace NUMINAMATH_CALUDE_division_by_fraction_problem_solution_l3734_373455

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b :=
by sorry

theorem problem_solution : (5 : ℚ) / ((7 : ℚ) / 13) = 65 / 7 :=
by sorry

end NUMINAMATH_CALUDE_division_by_fraction_problem_solution_l3734_373455


namespace NUMINAMATH_CALUDE_mel_katherine_age_difference_l3734_373494

/-- Given that Mel is younger than Katherine, and when Katherine is 24, Mel is 21,
    prove that Mel is 3 years younger than Katherine. -/
theorem mel_katherine_age_difference :
  ∀ (katherine_age mel_age : ℕ),
  katherine_age > mel_age →
  (katherine_age = 24 → mel_age = 21) →
  katherine_age - mel_age = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mel_katherine_age_difference_l3734_373494


namespace NUMINAMATH_CALUDE_invertible_elements_mod_8_l3734_373423

theorem invertible_elements_mod_8 :
  ∀ a : ℤ, a ∈ ({1, 3, 5, 7} : Set ℤ) ↔
    (∃ b : ℤ, (a * b) % 8 = 1 ∧ (a * a) % 8 = 1) :=
by sorry

end NUMINAMATH_CALUDE_invertible_elements_mod_8_l3734_373423


namespace NUMINAMATH_CALUDE_min_dot_product_l3734_373407

/-- Circle C with center (t, t-2) and radius 1 -/
def Circle (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - t)^2 + (p.2 - (t-2))^2 = 1}

/-- Point P -/
def P : ℝ × ℝ := (-1, 1)

/-- Tangent points A and B (existence assumed) -/
def TangentPoints (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Dot product of vectors PA and PB -/
def DotProduct (t : ℝ) : ℝ :=
  let (A, B) := TangentPoints t
  ((A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2))

theorem min_dot_product :
  ∃ (m : ℝ), (∀ t, DotProduct t ≥ m) ∧ (∃ t₀, DotProduct t₀ = m) ∧ m = 21/4 := by
  sorry

end NUMINAMATH_CALUDE_min_dot_product_l3734_373407


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_arithmetic_sequence_l3734_373475

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 8th term of the arithmetic sequence with first term 1 and common difference 3 is 22 -/
theorem eighth_term_of_specific_arithmetic_sequence :
  arithmeticSequenceTerm 1 3 8 = 22 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_arithmetic_sequence_l3734_373475


namespace NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l3734_373433

theorem cubic_equation_sum_of_cubes :
  ∃ (u v w : ℝ),
    (u - Real.rpow 7 (1/3 : ℝ)) * (u - Real.rpow 29 (1/3 : ℝ)) * (u - Real.rpow 61 (1/3 : ℝ)) = 1/5 ∧
    (v - Real.rpow 7 (1/3 : ℝ)) * (v - Real.rpow 29 (1/3 : ℝ)) * (v - Real.rpow 61 (1/3 : ℝ)) = 1/5 ∧
    (w - Real.rpow 7 (1/3 : ℝ)) * (w - Real.rpow 29 (1/3 : ℝ)) * (w - Real.rpow 61 (1/3 : ℝ)) = 1/5 ∧
    u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    u^3 + v^3 + w^3 = 97.6 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l3734_373433


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3734_373469

/-- Proves that given a principal of 5000, if increasing the interest rate by 3%
    results in 300 more interest over the same time period, then the time period is 2 years. -/
theorem simple_interest_problem (R : ℚ) (T : ℚ) : 
  (5000 * (R + 3) / 100 * T = 5000 * R / 100 * T + 300) → T = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3734_373469


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3734_373483

theorem arithmetic_expression_equality : 9 - 8 + 7 * 6 + 5 - 4 * 3 + 2 - 1 = 37 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3734_373483


namespace NUMINAMATH_CALUDE_painting_areas_l3734_373482

/-- Represents the areas painted in square decimeters -/
structure PaintedAreas where
  blue : ℝ
  green : ℝ
  yellow : ℝ

/-- The total amount of each paint color available in square decimeters -/
def total_paint : ℝ := 38

/-- Theorem stating the correct areas given the painting conditions -/
theorem painting_areas : ∃ (areas : PaintedAreas),
  -- All paint is used
  areas.blue + areas.yellow + areas.green = 2 * total_paint ∧
  -- Green paint mixture ratio
  areas.green = (2 * areas.yellow + areas.blue) / 3 ∧
  -- Grass area is 6 more than sky area
  areas.green = areas.blue + 6 ∧
  -- Correct areas
  areas.blue = 27 ∧
  areas.green = 33 ∧
  areas.yellow = 16 := by
  sorry

end NUMINAMATH_CALUDE_painting_areas_l3734_373482


namespace NUMINAMATH_CALUDE_cake_slices_l3734_373410

/-- The cost of ingredients and number of slices eaten by Laura's mother and the dog --/
structure CakeData where
  flour_cost : ℝ
  sugar_cost : ℝ
  butter_cost : ℝ
  eggs_cost : ℝ
  mother_slices : ℕ
  dog_cost : ℝ

/-- The total number of slices Laura cut the cake into --/
def total_slices (data : CakeData) : ℕ :=
  sorry

/-- Theorem stating that the total number of slices is 6 --/
theorem cake_slices (data : CakeData) 
  (h1 : data.flour_cost = 4)
  (h2 : data.sugar_cost = 2)
  (h3 : data.butter_cost = 2.5)
  (h4 : data.eggs_cost = 0.5)
  (h5 : data.mother_slices = 2)
  (h6 : data.dog_cost = 6) :
  total_slices data = 6 :=
sorry

end NUMINAMATH_CALUDE_cake_slices_l3734_373410


namespace NUMINAMATH_CALUDE_regular_price_correct_l3734_373442

/-- The regular price of one tire -/
def regular_price : ℝ := 108

/-- The sale price for three tires -/
def sale_price : ℝ := 270

/-- The theorem stating that the regular price of one tire is correct given the sale conditions -/
theorem regular_price_correct : 
  2 * regular_price + regular_price / 2 = sale_price :=
by sorry

end NUMINAMATH_CALUDE_regular_price_correct_l3734_373442


namespace NUMINAMATH_CALUDE_sum_of_four_triangles_l3734_373458

/-- The value of a square -/
def square_value : ℝ := sorry

/-- The value of a triangle -/
def triangle_value : ℝ := sorry

/-- All squares have the same value -/
axiom square_constant : ∀ s : ℝ, s = square_value

/-- All triangles have the same value -/
axiom triangle_constant : ∀ t : ℝ, t = triangle_value

/-- First equation: square + triangle + square + triangle + square = 27 -/
axiom equation_1 : 3 * square_value + 2 * triangle_value = 27

/-- Second equation: triangle + square + triangle + square + triangle = 23 -/
axiom equation_2 : 2 * square_value + 3 * triangle_value = 23

/-- The sum of four triangles equals 12 -/
theorem sum_of_four_triangles : 4 * triangle_value = 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_four_triangles_l3734_373458


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3734_373432

theorem tan_alpha_value (α : Real) (h : Real.tan α = 3/4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64/25 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3734_373432


namespace NUMINAMATH_CALUDE_no_prime_base_n_representation_l3734_373426

def base_n_representation (n : ℕ) : ℕ := n^4 + n^2 + 1

theorem no_prime_base_n_representation :
  ∀ n : ℕ, n ≥ 2 → ¬(Nat.Prime (base_n_representation n)) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_base_n_representation_l3734_373426
