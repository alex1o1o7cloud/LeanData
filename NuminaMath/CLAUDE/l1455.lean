import Mathlib

namespace orange_eaters_difference_l1455_145589

def family_gathering (total : ℕ) (orange_eaters : ℕ) (banana_eaters : ℕ) (apple_eaters : ℕ) : Prop :=
  total = 20 ∧
  orange_eaters = total / 2 ∧
  banana_eaters = (total - orange_eaters) / 2 ∧
  apple_eaters = total - orange_eaters - banana_eaters ∧
  orange_eaters < total

theorem orange_eaters_difference (total orange_eaters banana_eaters apple_eaters : ℕ) :
  family_gathering total orange_eaters banana_eaters apple_eaters →
  total - orange_eaters = 10 := by
  sorry

end orange_eaters_difference_l1455_145589


namespace wine_drinking_time_is_correct_l1455_145560

/-- Represents the time taken for three assistants to drink 40 liters of wine -/
def wine_drinking_time : ℚ :=
  let rate1 := (40 : ℚ) / 12  -- Rate of the first assistant
  let rate2 := (40 : ℚ) / 10  -- Rate of the second assistant
  let rate3 := (40 : ℚ) / 8   -- Rate of the third assistant
  let total_rate := rate1 + rate2 + rate3
  (40 : ℚ) / total_rate

/-- The wine drinking time is equal to 3 9/37 hours -/
theorem wine_drinking_time_is_correct : wine_drinking_time = 3 + 9 / 37 := by
  sorry

#eval wine_drinking_time

end wine_drinking_time_is_correct_l1455_145560


namespace cos_x_plus_2y_equals_one_l1455_145516

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4)) 
  (h2 : y ∈ Set.Icc (-π/4) (π/4)) 
  (h3 : x^3 + Real.sin x - 2*a = 0) 
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) : 
  Real.cos (x + 2*y) = 1 := by
sorry

end cos_x_plus_2y_equals_one_l1455_145516


namespace election_vote_count_l1455_145574

theorem election_vote_count 
  (candidate1_percentage : ℝ) 
  (candidate2_votes : ℕ) 
  (total_votes : ℕ) : 
  candidate1_percentage = 0.7 →
  candidate2_votes = 240 →
  (candidate2_votes : ℝ) / total_votes = 1 - candidate1_percentage →
  total_votes = 800 := by
sorry

end election_vote_count_l1455_145574


namespace exists_fibonacci_divisible_by_n_l1455_145542

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Statement of the theorem
theorem exists_fibonacci_divisible_by_n (n : ℕ) (hn : n > 0) : 
  ∃ m : ℕ, m > 0 ∧ n ∣ fib m :=
sorry

end exists_fibonacci_divisible_by_n_l1455_145542


namespace sector_max_area_l1455_145533

/-- Given a sector with perimeter 16, its maximum area is 16. -/
theorem sector_max_area (r θ : ℝ) (h : 2 * r + r * θ = 16) : 
  ∀ r' θ' : ℝ, 2 * r' + r' * θ' = 16 → (1/2) * r' * r' * θ' ≤ 16 := by
sorry

end sector_max_area_l1455_145533


namespace workers_total_earnings_l1455_145544

/-- Calculates the total earnings of three workers given their daily wages and days worked. -/
def total_earnings (daily_wage_a daily_wage_b daily_wage_c : ℕ) (days_a days_b days_c : ℕ) : ℕ :=
  daily_wage_a * days_a + daily_wage_b * days_b + daily_wage_c * days_c

/-- Theorem stating that under the given conditions, the total earnings of three workers is 1480. -/
theorem workers_total_earnings :
  ∀ (daily_wage_a daily_wage_b daily_wage_c : ℕ),
    daily_wage_a * 3 = daily_wage_b * 3 * 3/4 →
    daily_wage_b * 4 = daily_wage_c * 4 * 4/5 →
    daily_wage_c = 100 →
    total_earnings daily_wage_a daily_wage_b daily_wage_c 6 9 4 = 1480 :=
by
  sorry

#eval total_earnings 60 80 100 6 9 4

end workers_total_earnings_l1455_145544


namespace count_repetitive_permutations_formula_l1455_145536

/-- The count of n-repetitive permutations formed by a₁, a₂, a₃, a₄, a₅, a₆ 
    where both a₁ and a₃ each appear an even number of times -/
def count_repetitive_permutations (n : ℕ) : ℕ :=
  (6^n - 2 * 5^n + 4^n) / 4

/-- Theorem stating that the count of n-repetitive permutations with the given conditions
    is equal to (6^n - 2 * 5^n + 4^n) / 4 -/
theorem count_repetitive_permutations_formula (n : ℕ) :
  count_repetitive_permutations n = (6^n - 2 * 5^n + 4^n) / 4 := by
  sorry

end count_repetitive_permutations_formula_l1455_145536


namespace production_time_is_13_hours_l1455_145584

/-- The time needed to complete the remaining production task -/
def time_to_complete (total_parts : ℕ) (apprentice_rate : ℕ) (master_rate : ℕ) (parts_done : ℕ) : ℚ :=
  (total_parts - parts_done) / (apprentice_rate + master_rate)

/-- Proof that the time to complete the production task is 13 hours -/
theorem production_time_is_13_hours :
  let total_parts : ℕ := 500
  let apprentice_rate : ℕ := 15
  let master_rate : ℕ := 20
  let parts_done : ℕ := 45
  time_to_complete total_parts apprentice_rate master_rate parts_done = 13 := by
  sorry

#eval time_to_complete 500 15 20 45

end production_time_is_13_hours_l1455_145584


namespace product_of_recurring_decimal_and_nine_l1455_145514

theorem product_of_recurring_decimal_and_nine (x : ℚ) : 
  x = 1/3 → x * 9 = 3 := by
  sorry

end product_of_recurring_decimal_and_nine_l1455_145514


namespace trigonometric_equality_l1455_145562

theorem trigonometric_equality (x y : ℝ) 
  (h : (Real.sin x ^ 2 - Real.cos x ^ 2 + Real.cos x ^ 2 * Real.cos y ^ 2 - Real.sin x ^ 2 * Real.sin y ^ 2) / Real.sin (x + y) = 1) :
  ∃ k : ℤ, x - y = 2 * k * Real.pi + Real.pi / 2 := by
sorry

end trigonometric_equality_l1455_145562


namespace m_values_l1455_145585

def A (m : ℝ) : Set ℝ := {1, 2, 3, m}
def B (m : ℝ) : Set ℝ := {m^2, 3}

theorem m_values (m : ℝ) :
  A m ∪ B m = A m →
  m = -1 ∨ m = Real.sqrt 2 ∨ m = -Real.sqrt 2 ∨ m = 0 := by
  sorry

end m_values_l1455_145585


namespace yard_length_with_26_trees_l1455_145551

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1 : ℝ) * tree_distance

theorem yard_length_with_26_trees :
  yard_length 26 11 = 275 := by
  sorry

end yard_length_with_26_trees_l1455_145551


namespace worker_b_completion_time_l1455_145554

/-- Given a piece of work that can be completed by three workers a, b, and c, 
    this theorem proves the time taken by worker b to complete the work alone. -/
theorem worker_b_completion_time 
  (total_time : ℝ) 
  (time_a : ℝ) 
  (time_c : ℝ) 
  (h1 : total_time = 4) 
  (h2 : time_a = 36) 
  (h3 : time_c = 6) : 
  ∃ (time_b : ℝ), time_b = 18 ∧ 
  1 / total_time = 1 / time_a + 1 / time_b + 1 / time_c :=
by sorry

end worker_b_completion_time_l1455_145554


namespace chickens_and_rabbits_l1455_145588

theorem chickens_and_rabbits (total_heads total_feet : ℕ) 
  (h1 : total_heads = 35) 
  (h2 : total_feet = 94) : 
  ∃ (chickens rabbits : ℕ), 
    chickens + rabbits = total_heads ∧ 
    2 * chickens + 4 * rabbits = total_feet ∧ 
    chickens = 23 ∧ 
    rabbits = 12 := by
  sorry

#check chickens_and_rabbits

end chickens_and_rabbits_l1455_145588


namespace box_surface_area_is_1600_l1455_145543

/-- Calculates the surface area of the interior of an open box formed by removing square corners from a rectangular sheet and folding the sides. -/
def boxSurfaceArea (length width cornerSize : ℕ) : ℕ :=
  length * width - 4 * (cornerSize * cornerSize)

/-- Theorem stating that the surface area of the interior of the box is 1600 square units. -/
theorem box_surface_area_is_1600 :
  boxSurfaceArea 40 50 10 = 1600 := by
  sorry

#eval boxSurfaceArea 40 50 10

end box_surface_area_is_1600_l1455_145543


namespace sara_has_108_golf_balls_l1455_145531

/-- The number of dozens of golf balls Sara has -/
def saras_dozens : ℕ := 9

/-- The number of items in one dozen -/
def items_per_dozen : ℕ := 12

/-- The total number of golf balls Sara has -/
def saras_golf_balls : ℕ := saras_dozens * items_per_dozen

theorem sara_has_108_golf_balls : saras_golf_balls = 108 := by
  sorry

end sara_has_108_golf_balls_l1455_145531


namespace swift_stream_pump_l1455_145519

/-- The SwiftStream pump problem -/
theorem swift_stream_pump (pump_rate : ℝ) (time : ℝ) (volume : ℝ) : 
  pump_rate = 500 → time = 1/2 → volume = pump_rate * time → volume = 250 := by
  sorry

end swift_stream_pump_l1455_145519


namespace square_plus_reciprocal_square_l1455_145540

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end square_plus_reciprocal_square_l1455_145540


namespace school_students_count_l1455_145547

/-- Represents the donation and student information for a school --/
structure SchoolDonation where
  total_donation : ℕ
  average_donation_7_8 : ℕ
  grade_9_intended_donation : ℕ
  grade_9_rejection_rate : ℚ

/-- Calculates the total number of students in the school based on donation information --/
def total_students (sd : SchoolDonation) : ℕ :=
  sd.total_donation / sd.average_donation_7_8

/-- Theorem stating that the total number of students in the school is 224 --/
theorem school_students_count (sd : SchoolDonation) 
  (h1 : sd.total_donation = 13440)
  (h2 : sd.average_donation_7_8 = 60)
  (h3 : sd.grade_9_intended_donation = 100)
  (h4 : sd.grade_9_rejection_rate = 2/5) :
  total_students sd = 224 := by
  sorry

#eval total_students { 
  total_donation := 13440, 
  average_donation_7_8 := 60, 
  grade_9_intended_donation := 100, 
  grade_9_rejection_rate := 2/5 
}

end school_students_count_l1455_145547


namespace d_squared_plus_5d_l1455_145556

theorem d_squared_plus_5d (d : ℤ) : d = 5 + 6 → d^2 + 5*d = 176 := by
  sorry

end d_squared_plus_5d_l1455_145556


namespace exists_unique_decomposition_l1455_145508

def sequence_decomposition (x : ℕ → ℝ) : Prop :=
  ∃! (y z : ℕ → ℝ), 
    (∀ n : ℕ, x n = y n - z n) ∧
    (∀ n : ℕ, y n ≥ 0) ∧
    (∀ n : ℕ, n > 0 → z n ≥ z (n-1)) ∧
    (∀ n : ℕ, n > 0 → y n * (z n - z (n-1)) = 0) ∧
    (z 0 = 0)

theorem exists_unique_decomposition (x : ℕ → ℝ) : sequence_decomposition x := by
  sorry

end exists_unique_decomposition_l1455_145508


namespace distance_product_range_l1455_145553

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x
def C₂ (x y : ℝ) : Prop := (x-4)^2 + y^2 = 8

-- Define a point P on C₁
structure PointOnC₁ where
  x : ℝ
  y : ℝ
  on_C₁ : C₁ x y

-- Define the line l with 45° inclination passing through P
def line_l (P : PointOnC₁) (x y : ℝ) : Prop :=
  y - P.y = (x - P.x)

-- Define the intersection points Q and R
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_C₂ : C₂ x y
  on_l : line_l P x y

-- Define the product of distances |PQ| · |PR|
def distance_product (P : PointOnC₁) (Q R : IntersectionPoint) : ℝ :=
  ((Q.x - P.x)^2 + (Q.y - P.y)^2) * ((R.x - P.x)^2 + (R.y - P.y)^2)

-- State the theorem
theorem distance_product_range (P : PointOnC₁) (Q R : IntersectionPoint) 
  (h_distinct : Q ≠ R) :
  ∃ (d : ℝ), distance_product P Q R = d ∧ (d ∈ Set.Icc 4 8 ∨ d ∈ Set.Ioo 8 200) :=
sorry

end distance_product_range_l1455_145553


namespace reese_practice_hours_l1455_145535

/-- Calculates the total piano practice hours for Reese over a given number of months -/
def total_practice_hours (months : ℕ) : ℕ :=
  let initial_weekly_hours := 4
  let initial_months := 2
  let increased_weekly_hours := 5
  let workshop_hours := 3
  
  let initial_practice := initial_weekly_hours * 4 * min months initial_months
  let increased_practice := increased_weekly_hours * 4 * max (months - initial_months) 0
  let total_workshops := months * workshop_hours
  
  initial_practice + increased_practice + total_workshops

/-- Theorem stating that Reese's total practice hours after 5 months is 107 -/
theorem reese_practice_hours : total_practice_hours 5 = 107 := by
  sorry

end reese_practice_hours_l1455_145535


namespace floor_equation_solution_l1455_145511

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ - 1⌋ = ⌊x + 3⌋) ↔ (5/3 ≤ x ∧ x < 3 ∧ x ≠ 2) :=
sorry

end floor_equation_solution_l1455_145511


namespace not_cheap_necessary_not_sufficient_for_good_quality_l1455_145570

-- Define the universe of products
variable (Product : Type)

-- Define predicates for product qualities
variable (not_cheap : Product → Prop)
variable (good_quality : Product → Prop)

-- Define the saying "you get what you pay for" as an axiom
axiom you_get_what_you_pay_for : ∀ (p : Product), good_quality p → not_cheap p

-- Theorem to prove
theorem not_cheap_necessary_not_sufficient_for_good_quality :
  (∀ (p : Product), good_quality p → not_cheap p) ∧
  (∃ (p : Product), not_cheap p ∧ ¬good_quality p) :=
sorry

end not_cheap_necessary_not_sufficient_for_good_quality_l1455_145570


namespace jennifer_additional_tanks_l1455_145527

/-- Represents the number of fish in each type of tank --/
structure TankCapacity where
  goldfish : Nat
  betta : Nat
  guppy : Nat
  clownfish : Nat

/-- Represents the number of tanks for each type of fish --/
structure TankCount where
  goldfish : Nat
  betta : Nat
  guppy : Nat
  clownfish : Nat

/-- Calculates the total number of fish given tank capacities and counts --/
def totalFish (capacity : TankCapacity) (count : TankCount) : Nat :=
  capacity.goldfish * count.goldfish +
  capacity.betta * count.betta +
  capacity.guppy * count.guppy +
  capacity.clownfish * count.clownfish

/-- Calculates the total number of tanks --/
def totalTanks (count : TankCount) : Nat :=
  count.goldfish + count.betta + count.guppy + count.clownfish

/-- Represents Jennifer's aquarium setup --/
def jennifer_setup : Prop :=
  ∃ (capacity : TankCapacity) (existing_count : TankCount) (new_count : TankCount),
    capacity.goldfish = 15 ∧
    capacity.betta = 1 ∧
    capacity.guppy = 5 ∧
    capacity.clownfish = 4 ∧
    existing_count.goldfish = 3 ∧
    existing_count.betta = 0 ∧
    existing_count.guppy = 0 ∧
    existing_count.clownfish = 0 ∧
    totalFish capacity (TankCount.mk
      existing_count.goldfish
      (existing_count.betta + new_count.betta)
      (existing_count.guppy + new_count.guppy)
      (existing_count.clownfish + new_count.clownfish)) = 75 ∧
    new_count.betta + new_count.guppy + new_count.clownfish = 15 ∧
    ∀ (alt_count : TankCount),
      totalFish capacity (TankCount.mk
        existing_count.goldfish
        (existing_count.betta + alt_count.betta)
        (existing_count.guppy + alt_count.guppy)
        (existing_count.clownfish + alt_count.clownfish)) = 75 →
      totalTanks alt_count ≥ totalTanks new_count

theorem jennifer_additional_tanks : jennifer_setup := by
  sorry

end jennifer_additional_tanks_l1455_145527


namespace class_vision_median_l1455_145517

/-- Represents the vision data for a class of students -/
structure VisionData where
  visions : List ℝ
  counts : List ℕ
  total_students : ℕ

/-- Calculates the median of a VisionData set -/
def median (data : VisionData) : ℝ :=
  sorry

/-- The specific vision data for the class -/
def class_vision_data : VisionData :=
  { visions := [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    counts := [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5],
    total_students := 39 }

/-- Theorem stating that the median of the class vision data is 4.6 -/
theorem class_vision_median :
  median class_vision_data = 4.6 := by
  sorry

end class_vision_median_l1455_145517


namespace min_value_expression_equality_condition_l1455_145500

theorem min_value_expression (y : ℝ) :
  y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) ≥ 1/27 :=
sorry

theorem equality_condition :
  ∃ y : ℝ, y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) = 1/27 :=
sorry

end min_value_expression_equality_condition_l1455_145500


namespace wire_service_reporters_l1455_145596

theorem wire_service_reporters (total : ℝ) (local_politics : ℝ) (non_local_politics : ℝ) 
  (h1 : local_politics = 0.18 * total)
  (h2 : non_local_politics = 0.4 * (local_politics + non_local_politics)) :
  (total - (local_politics + non_local_politics)) / total = 0.7 := by
  sorry

end wire_service_reporters_l1455_145596


namespace ice_cream_melt_time_l1455_145549

/-- The time it takes for an ice cream cone to melt, given the distance to the beach and Jack's jogging speed -/
theorem ice_cream_melt_time 
  (blocks_to_beach : ℕ)
  (miles_per_block : ℚ)
  (jogging_speed : ℚ)
  (h1 : blocks_to_beach = 16)
  (h2 : miles_per_block = 1 / 8)
  (h3 : jogging_speed = 12) :
  (blocks_to_beach : ℚ) * miles_per_block / jogging_speed * 60 = 10 := by
  sorry

#check ice_cream_melt_time

end ice_cream_melt_time_l1455_145549


namespace circle_packing_problem_l1455_145597

theorem circle_packing_problem (n : ℕ) :
  (n^2 = ((n + 14) * (n + 15)) / 2) → n^2 = 1225 := by
  sorry

end circle_packing_problem_l1455_145597


namespace equation_solution_inequality_system_solution_l1455_145591

-- Part 1: Equation solution
theorem equation_solution :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ 0 → (x / (x - 1) = (x - 1) / (2*x - 2) ↔ x = -1) :=
sorry

-- Part 2: Inequality system solution
theorem inequality_system_solution :
  ∀ x : ℝ, (5*x - 1 > 3*x - 4 ∧ -1/3*x ≤ 2/3 - x) ↔ (-3/2 < x ∧ x ≤ 1) :=
sorry

end equation_solution_inequality_system_solution_l1455_145591


namespace valid_lineup_count_is_14_l1455_145550

/-- Represents the four athletes in the relay race -/
inductive Athlete : Type
| A : Athlete
| B : Athlete
| C : Athlete
| D : Athlete

/-- Represents the four positions in the relay race -/
inductive Position : Type
| first : Position
| second : Position
| third : Position
| fourth : Position

/-- A valid lineup for the relay race -/
def Lineup := Position → Athlete

/-- Predicate to check if a lineup is valid according to the given conditions -/
def isValidLineup (l : Lineup) : Prop :=
  l Position.first ≠ Athlete.A ∧ l Position.fourth ≠ Athlete.B

/-- The number of valid lineups -/
def validLineupCount : ℕ := sorry

/-- Theorem stating that the number of valid lineups is 14 -/
theorem valid_lineup_count_is_14 : validLineupCount = 14 := by sorry

end valid_lineup_count_is_14_l1455_145550


namespace multiple_of_twelve_l1455_145530

theorem multiple_of_twelve (x : ℤ) : 
  (∃ k : ℤ, 7 * x - 3 = 12 * k) ↔ 
  (∃ t : ℤ, x = 12 * t + 9 ∨ x = 12 * t + 1029) :=
sorry

end multiple_of_twelve_l1455_145530


namespace expression_evaluation_l1455_145575

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := -2
  (3*x - 2*y)^2 - (2*y + x)*(2*y - x) - 2*x*(5*x - 6*y + x*y) = 1 :=
by sorry

end expression_evaluation_l1455_145575


namespace shelter_dogs_l1455_145590

theorem shelter_dogs (C : ℕ) (h1 : C > 0) (h2 : (15 : ℚ) / C = 11 / (C + 8)) : 
  (15 : ℕ) * C = 15 * 15 :=
sorry

end shelter_dogs_l1455_145590


namespace perpendicular_vector_t_value_l1455_145504

/-- Given vectors a and b, if a is perpendicular to (t*a + b), then t = -5 -/
theorem perpendicular_vector_t_value (a b : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (1, -1))
  (h2 : b = (6, -4))
  (h3 : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) :
  t = -5 := by sorry

end perpendicular_vector_t_value_l1455_145504


namespace timmy_calories_needed_l1455_145520

/-- Represents the number of calories in an orange -/
def calories_per_orange : ℕ := 80

/-- Represents the cost of an orange in cents -/
def cost_per_orange : ℕ := 120

/-- Represents Timmy's initial amount of money in cents -/
def initial_money : ℕ := 1000

/-- Represents the amount of money Timmy has left after buying oranges in cents -/
def money_left : ℕ := 400

/-- Calculates the number of calories Timmy needs to get -/
def calories_needed : ℕ := 
  ((initial_money - money_left) / cost_per_orange) * calories_per_orange

theorem timmy_calories_needed : calories_needed = 400 := by
  sorry

end timmy_calories_needed_l1455_145520


namespace problem_solution_l1455_145561

theorem problem_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b * c = 1) (h5 : a + 1 / c = 7) (h6 : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 := by
sorry

end problem_solution_l1455_145561


namespace danielle_apartment_rooms_l1455_145552

theorem danielle_apartment_rooms : 
  ∀ (heidi grant danielle : ℕ),
  heidi = 3 * danielle →
  grant * 9 = heidi →
  grant = 2 →
  danielle = 6 := by
sorry

end danielle_apartment_rooms_l1455_145552


namespace geometric_sequence_problem_l1455_145537

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) →  -- geometric sequence
  a 2 = 1 →  -- a_2 = 1
  a 8 = a 6 + 6 * a 4 →  -- a_8 = a_6 + 6a_4
  a 3 = Real.sqrt 3 :=  -- a_3 = √3
by sorry

end geometric_sequence_problem_l1455_145537


namespace electronic_product_failure_probability_l1455_145563

theorem electronic_product_failure_probability
  (p_working : ℝ)
  (h_working : p_working = 0.992)
  (h_probability : 0 ≤ p_working ∧ p_working ≤ 1) :
  1 - p_working = 0.008 := by
sorry

end electronic_product_failure_probability_l1455_145563


namespace average_speed_calculation_l1455_145506

theorem average_speed_calculation (speed1 speed2 : ℝ) (h1 : speed1 = 70) (h2 : speed2 = 90) :
  (speed1 + speed2) / 2 = 80 := by
  sorry

end average_speed_calculation_l1455_145506


namespace watson_class_second_graders_l1455_145521

/-- The number of second graders in Ms. Watson's class -/
def second_graders (kindergartners first_graders total_students : ℕ) : ℕ :=
  total_students - (kindergartners + first_graders)

/-- Theorem stating the number of second graders in Ms. Watson's class -/
theorem watson_class_second_graders :
  second_graders 14 24 42 = 4 := by
  sorry

end watson_class_second_graders_l1455_145521


namespace special_die_probability_sum_l1455_145501

/-- Represents a die with special probability distribution -/
structure SpecialDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℚ
  /-- Probability of rolling an even number -/
  even_prob : ℚ
  /-- Ensure even probability is twice odd probability -/
  even_twice_odd : even_prob = 2 * odd_prob
  /-- Ensure total probability is 1 -/
  total_prob_one : 3 * odd_prob + 3 * even_prob = 1

/-- Calculates the probability of rolling 1, 2, or 3 on the special die -/
def prob_not_exceeding_three (d : SpecialDie) : ℚ :=
  2 * d.odd_prob + d.even_prob

/-- The main theorem stating the sum of numerator and denominator is 13 -/
theorem special_die_probability_sum : 
  ∀ (d : SpecialDie), 
  let p := prob_not_exceeding_three d
  let n := p.den
  let m := p.num
  m + n = 13 := by sorry

end special_die_probability_sum_l1455_145501


namespace taxi_fare_proof_l1455_145595

/-- Proves that given an initial fare of $2.00 for the first 1/5 mile and a total fare of $25.40 for an 8-mile ride, the fare for each 1/5 mile after the first 1/5 mile is $0.60. -/
theorem taxi_fare_proof (initial_fare : ℝ) (total_fare : ℝ) (ride_distance : ℝ) 
  (h1 : initial_fare = 2)
  (h2 : total_fare = 25.4)
  (h3 : ride_distance = 8) :
  let increments : ℝ := ride_distance * 5
  let remaining_fare : ℝ := total_fare - initial_fare
  let remaining_increments : ℝ := increments - 1
  remaining_fare / remaining_increments = 0.6 := by
sorry

end taxi_fare_proof_l1455_145595


namespace max_value_of_sqrt_sum_max_value_achievable_l1455_145568

theorem max_value_of_sqrt_sum (x y z : ℝ) :
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 6 →
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 3 * Real.sqrt 20 :=
by sorry

theorem max_value_achievable (x y z : ℝ) :
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 6 →
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 6 ∧
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) = 3 * Real.sqrt 20 :=
by sorry

end max_value_of_sqrt_sum_max_value_achievable_l1455_145568


namespace sufficient_not_necessary_condition_l1455_145509

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≥ 5) → (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) → a ≥ 5) :=
by sorry

end sufficient_not_necessary_condition_l1455_145509


namespace binomial_12_choose_6_l1455_145528

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end binomial_12_choose_6_l1455_145528


namespace crayon_distribution_l1455_145586

theorem crayon_distribution (initial_boxes : Nat) (crayons_per_box : Nat) 
  (to_mae : Nat) (to_rey : Nat) (left : Nat) :
  initial_boxes = 7 →
  crayons_per_box = 15 →
  to_mae = 12 →
  to_rey = 20 →
  left = 25 →
  (initial_boxes * crayons_per_box - to_mae - to_rey - left) - to_mae = 36 := by
  sorry

end crayon_distribution_l1455_145586


namespace parallel_lines_k_value_l1455_145580

/-- Given two points on a line and another line equation, 
    prove that the value of k for which the lines are parallel is 14. -/
theorem parallel_lines_k_value (k : ℝ) : 
  (∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ 6 * x - 2 * y = -8) ∧
   (∃ (m' b' : ℝ), m' = m ∧ 23 = m' * k + b' ∧ -4 = m' * 5 + b')) →
  k = 14 := by
sorry

end parallel_lines_k_value_l1455_145580


namespace floor_negative_seven_thirds_l1455_145529

theorem floor_negative_seven_thirds : ⌊(-7 : ℚ) / 3⌋ = -3 := by
  sorry

end floor_negative_seven_thirds_l1455_145529


namespace min_value_with_constraint_l1455_145572

theorem min_value_with_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_xyz : x * y * z = 3) :
  x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 ≥ 162 ∧ 
  (x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 = 162 ↔ x = 3 ∧ y = 1/2 ∧ z = 2) :=
by sorry

end min_value_with_constraint_l1455_145572


namespace fishing_tournament_l1455_145502

theorem fishing_tournament (jacob_initial : ℕ) : 
  (7 * jacob_initial - 23 = jacob_initial + 26 - 1) → jacob_initial = 8 := by sorry

end fishing_tournament_l1455_145502


namespace halfway_fraction_l1455_145569

theorem halfway_fraction (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) :
  (a + b) / 2 = 19/24 := by sorry

end halfway_fraction_l1455_145569


namespace quadratic_always_real_roots_roots_ratio_three_implies_m_values_l1455_145573

/-- The quadratic equation x^2 - 4x - m(m+4) = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 4*x - m*(m+4) = 0

theorem quadratic_always_real_roots :
  ∀ m : ℝ, ∃ x₁ x₂ : ℝ, quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ x₁ ≠ x₂ :=
sorry

theorem roots_ratio_three_implies_m_values :
  ∀ m x₁ x₂ : ℝ, quadratic_equation x₁ m → quadratic_equation x₂ m → x₂ = 3*x₁ →
  m = -1 ∨ m = -3 :=
sorry

end quadratic_always_real_roots_roots_ratio_three_implies_m_values_l1455_145573


namespace simplify_nested_expression_l1455_145555

theorem simplify_nested_expression (x : ℝ) : 1 - (2 - (2 - (2 - (2 - x)))) = 1 - x := by
  sorry

end simplify_nested_expression_l1455_145555


namespace pie_difference_l1455_145564

/-- The number of apple pies baked on Mondays and Fridays -/
def monday_friday_apple : ℕ := 16

/-- The number of apple pies baked on Wednesdays -/
def wednesday_apple : ℕ := 20

/-- The number of cherry pies baked on Tuesdays -/
def tuesday_cherry : ℕ := 14

/-- The number of cherry pies baked on Thursdays -/
def thursday_cherry : ℕ := 18

/-- The number of apple pies baked on Saturdays -/
def saturday_apple : ℕ := 10

/-- The number of cherry pies baked on Saturdays -/
def saturday_cherry : ℕ := 8

/-- The number of apple pies baked on Sundays -/
def sunday_apple : ℕ := 6

/-- The number of cherry pies baked on Sundays -/
def sunday_cherry : ℕ := 12

/-- The total number of apple pies baked in one week -/
def total_apple : ℕ := 2 * monday_friday_apple + wednesday_apple + saturday_apple + sunday_apple

/-- The total number of cherry pies baked in one week -/
def total_cherry : ℕ := tuesday_cherry + thursday_cherry + saturday_cherry + sunday_cherry

theorem pie_difference : total_apple - total_cherry = 16 := by
  sorry

end pie_difference_l1455_145564


namespace imaginary_equation_solution_l1455_145566

theorem imaginary_equation_solution (z : ℂ) (b : ℝ) : 
  (z.re = 0) →  -- z is a pure imaginary number
  ((2 - I) * z = 4 - b * (1 + I)^2) →
  b = -4 :=
by sorry

end imaginary_equation_solution_l1455_145566


namespace both_selected_probability_l1455_145571

theorem both_selected_probability 
  (ram_prob : ℚ) 
  (ravi_prob : ℚ) 
  (h1 : ram_prob = 6 / 7) 
  (h2 : ravi_prob = 1 / 5) : 
  ram_prob * ravi_prob = 6 / 35 := by
sorry

end both_selected_probability_l1455_145571


namespace book_arrangement_count_l1455_145567

/-- Represents the number of books -/
def n : ℕ := 6

/-- Represents the number of ways to arrange books A and B at the ends -/
def end_arrangements : ℕ := 2

/-- Represents the number of ways to order books C and D -/
def cd_orders : ℕ := 2

/-- Represents the number of ways to arrange the C-D pair and the other 2 books in the middle -/
def middle_arrangements : ℕ := 6

/-- The total number of valid arrangements -/
def total_arrangements : ℕ := end_arrangements * cd_orders * middle_arrangements

theorem book_arrangement_count :
  total_arrangements = 24 :=
sorry

end book_arrangement_count_l1455_145567


namespace dice_probability_theorem_l1455_145513

/-- Represents a 12-sided die with colored sides -/
structure ColoredDie :=
  (violet : ℕ)
  (orange : ℕ)
  (lime : ℕ)
  (total : ℕ)
  (h1 : violet + orange + lime = total)
  (h2 : total = 12)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.violet^2 + d.orange^2 + d.lime^2) / d.total^2

/-- Theorem statement for the probability problem -/
theorem dice_probability_theorem (d : ColoredDie) 
  (hv : d.violet = 3) (ho : d.orange = 4) (hl : d.lime = 5) : 
  same_color_probability d = 25 / 72 := by
  sorry

#eval same_color_probability ⟨3, 4, 5, 12, by norm_num, rfl⟩

end dice_probability_theorem_l1455_145513


namespace total_viewing_time_l1455_145505

-- Define the viewing segments
def segment1 : ℕ := 35
def segment2 : ℕ := 45
def segment3 : ℕ := 20

-- Define the rewind times
def rewind1 : ℕ := 5
def rewind2 : ℕ := 15

-- Theorem to prove
theorem total_viewing_time :
  segment1 + segment2 + segment3 + rewind1 + rewind2 = 120 := by
  sorry

end total_viewing_time_l1455_145505


namespace guard_max_demand_l1455_145576

/-- Represents the outcome of the outsider's decision -/
inductive Outcome
| Pay
| Refuse

/-- Represents the guard's demand and the outsider's decision -/
structure Scenario where
  guardDemand : ℕ
  outsiderDecision : Outcome

/-- Calculates the outsider's loss based on the scenario -/
def outsiderLoss (s : Scenario) : ℤ :=
  match s.outsiderDecision with
  | Outcome.Pay => s.guardDemand - 100
  | Outcome.Refuse => 100

/-- Determines if the outsider will pay based on personal benefit -/
def willPay (guardDemand : ℕ) : Prop :=
  outsiderLoss { guardDemand := guardDemand, outsiderDecision := Outcome.Pay } <
  outsiderLoss { guardDemand := guardDemand, outsiderDecision := Outcome.Refuse }

/-- The maximum number of coins the guard can demand -/
def maxGuardDemand : ℕ := 199

theorem guard_max_demand :
  (∀ n : ℕ, n ≤ maxGuardDemand → willPay n) ∧
  (∀ n : ℕ, n > maxGuardDemand → ¬willPay n) :=
sorry

end guard_max_demand_l1455_145576


namespace school_bus_capacity_l1455_145583

/-- The number of rows of seats in the school bus -/
def num_rows : ℕ := 20

/-- The number of kids that can sit in each row -/
def kids_per_row : ℕ := 4

/-- The total number of kids that can ride the school bus -/
def total_capacity : ℕ := num_rows * kids_per_row

theorem school_bus_capacity : total_capacity = 80 := by
  sorry

end school_bus_capacity_l1455_145583


namespace angle_b_range_in_geometric_progression_triangle_l1455_145599

/-- In a triangle ABC, if sides a, b, c form a geometric progression,
    then angle B is in the range (0, π/3] -/
theorem angle_b_range_in_geometric_progression_triangle
  (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  b^2 = a * c →
  0 < B ∧ B ≤ π/3 := by
  sorry

end angle_b_range_in_geometric_progression_triangle_l1455_145599


namespace six_digit_numbers_with_zero_l1455_145525

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers without any zero -/
def six_digit_numbers_without_zero : ℕ := 531441

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero :
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := by
  sorry

end six_digit_numbers_with_zero_l1455_145525


namespace right_triangle_area_l1455_145587

/-- The area of a right triangle with hypotenuse 9 inches and one angle 30° --/
theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) : 
  h = 9 →  -- hypotenuse is 9 inches
  α = 30 * π / 180 →  -- one angle is 30°
  area = (9^2 * Real.sin (30 * π / 180) * Real.sin (60 * π / 180)) / 4 →  -- area formula for right triangle
  area = 10.125 * Real.sqrt 3 := by
  sorry

end right_triangle_area_l1455_145587


namespace exponential_inequality_range_l1455_145578

theorem exponential_inequality_range (a : ℝ) :
  (∀ x : ℝ, Real.exp (2 * x) - (a - 3) * Real.exp x + 4 - 3 * a > 0) →
  a < 4/3 :=
by sorry

end exponential_inequality_range_l1455_145578


namespace recipe_ratio_l1455_145598

/-- Given a recipe with 5 cups of flour and 1 cup of shortening,
    if 2/3 cup of shortening is used, then 3 1/3 cups of flour
    should be used to maintain the same ratio. -/
theorem recipe_ratio (original_flour : ℚ) (original_shortening : ℚ)
                     (available_shortening : ℚ) (needed_flour : ℚ) :
  original_flour = 5 →
  original_shortening = 1 →
  available_shortening = 2/3 →
  needed_flour = 10/3 →
  needed_flour / available_shortening = original_flour / original_shortening :=
by sorry

end recipe_ratio_l1455_145598


namespace cindys_calculation_l1455_145503

theorem cindys_calculation (x : ℝ) : (x - 8) / 4 = 24 → (x - 4) / 8 = 12.5 := by
  sorry

end cindys_calculation_l1455_145503


namespace longest_rod_in_cube_l1455_145581

theorem longest_rod_in_cube (side_length : ℝ) (h : side_length = 4) :
  Real.sqrt (3 * side_length^2) = 4 * Real.sqrt 3 := by
  sorry

end longest_rod_in_cube_l1455_145581


namespace complex_equation_solution_l1455_145582

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2*Complex.I → z = -1 + (3/2)*Complex.I := by
  sorry

end complex_equation_solution_l1455_145582


namespace triangular_prism_area_bound_l1455_145538

/-- Given a triangular prism P-ABC with specified side lengths, 
    the sum of squared areas of triangles ABC and PBC is bounded. -/
theorem triangular_prism_area_bound 
  (AB : ℝ) (AC : ℝ) (PB : ℝ) (PC : ℝ)
  (h_AB : AB = Real.sqrt 3)
  (h_AC : AC = 1)
  (h_PB : PB = Real.sqrt 2)
  (h_PC : PC = Real.sqrt 2) :
  ∃ (S_ABC S_PBC : ℝ),
    (1/4 : ℝ) < S_ABC^2 + S_PBC^2 ∧ 
    S_ABC^2 + S_PBC^2 ≤ (7/4 : ℝ) :=
by sorry

end triangular_prism_area_bound_l1455_145538


namespace seven_divides_special_integer_l1455_145546

/-- Represents a 7-digit positive integer with the specified structure -/
structure SevenDigitInteger where
  value : ℕ
  is_seven_digit : 1000000 ≤ value ∧ value < 10000000
  first_three_equals_middle_three : ∃ (a b c : ℕ), value = a * 1000000 + b * 100000 + c * 10000 + a * 1000 + b * 100 + c * 10 + (value % 10)
  last_digit_multiple_of_first : ∃ (k : ℕ), value % 10 = k * ((value / 1000000) % 10)

/-- Theorem stating that 7 is a factor of any SevenDigitInteger -/
theorem seven_divides_special_integer (W : SevenDigitInteger) : 7 ∣ W.value := by
  sorry

end seven_divides_special_integer_l1455_145546


namespace quadratic_one_solution_l1455_145526

theorem quadratic_one_solution (p : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + p = 0) ↔ p = 49/12 := by
  sorry

end quadratic_one_solution_l1455_145526


namespace remainder_sum_l1455_145524

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 47) 
  (hd : d % 45 = 14) : 
  (c + d) % 15 = 1 := by
sorry

end remainder_sum_l1455_145524


namespace smallest_staircase_steps_l1455_145532

theorem smallest_staircase_steps : ∃ n : ℕ,
  n > 15 ∧
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 7 = 1 ∧
  (∀ m : ℕ, m > 15 ∧ m % 3 = 1 ∧ m % 5 = 3 ∧ m % 7 = 1 → m ≥ n) ∧
  n = 73 := by
sorry

end smallest_staircase_steps_l1455_145532


namespace special_line_equation_l1455_145541

/-- A line passing through (-4, -1) with x-intercept twice its y-intercept -/
structure SpecialLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (-4, -1) -/
  passes_through : slope * (-4) + y_intercept = -1
  /-- The x-intercept is twice the y-intercept -/
  intercept_relation : -y_intercept / slope = 2 * y_intercept

/-- The equation of the special line is x + 2y + 6 = 0 or y = 1/4 x -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + 2*y + 6 = 0) ∨
  (l.slope = 1/4 ∧ l.y_intercept = 0) :=
sorry

end special_line_equation_l1455_145541


namespace min_draw_for_red_card_l1455_145593

theorem min_draw_for_red_card (total : ℕ) (blue yellow red : ℕ) :
  total = 20 →
  blue + yellow + red = total →
  blue = yellow / 6 →
  red < yellow →
  15 = total - red + 1 :=
by sorry

end min_draw_for_red_card_l1455_145593


namespace negation_of_forall_greater_than_five_l1455_145548

theorem negation_of_forall_greater_than_five (S : Set ℝ) :
  (¬ ∀ x ∈ S, x > 5) ↔ (∃ x ∈ S, x ≤ 5) := by
  sorry

end negation_of_forall_greater_than_five_l1455_145548


namespace smallest_area_of_2020th_square_l1455_145539

theorem smallest_area_of_2020th_square (n : ℕ) (A : ℕ) : 
  n > 0 → 
  n^2 = 2019 + A → 
  A ≠ 1 → 
  (∀ m : ℕ, m > 0 ∧ m^2 = 2019 + A → n ≤ m) → 
  A ≥ 6 :=
by sorry

end smallest_area_of_2020th_square_l1455_145539


namespace complex_inequality_condition_l1455_145518

theorem complex_inequality_condition (z : ℂ) :
  (∀ z, Complex.abs z ≤ 1 → Complex.abs (Complex.re z) ≤ 1 ∧ Complex.abs (Complex.im z) ≤ 1) ∧
  (∃ z, Complex.abs (Complex.re z) ≤ 1 ∧ Complex.abs (Complex.im z) ≤ 1 ∧ Complex.abs z > 1) :=
by sorry

end complex_inequality_condition_l1455_145518


namespace system_solution_l1455_145565

theorem system_solution :
  ∀ x y z : ℝ,
  (x * y = z * (x + y + z) ∧
   y * z = 4 * x * (x + y + z) ∧
   z * x = 9 * y * (x + y + z)) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨
   ∃ t : ℝ, t ≠ 0 ∧ x = -3 * t ∧ y = -2 * t ∧ z = 6 * t) :=
by sorry

end system_solution_l1455_145565


namespace fifth_number_21st_row_l1455_145545

/-- Represents the triangular array of numbers -/
def TriangularArray (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem fifth_number_21st_row :
  (∀ n : ℕ, TriangularArray n n = n^2) →
  (∀ n k : ℕ, k < n → TriangularArray n (k+1) = TriangularArray n k + 1) →
  (∀ n : ℕ, TriangularArray (n+1) 1 = TriangularArray n n + 1) →
  TriangularArray 21 5 = 405 :=
sorry

end fifth_number_21st_row_l1455_145545


namespace find_set_N_l1455_145515

def U : Set ℕ := {1, 2, 3, 4, 5}

theorem find_set_N (M N : Set ℕ) 
  (h1 : U = M ∪ N) 
  (h2 : M ∩ (U \ N) = {2, 4}) : 
  N = {1, 3, 5} := by
  sorry

end find_set_N_l1455_145515


namespace tangent_intersections_symmetric_l1455_145522

/-- A line intersecting two hyperbolas -/
structure IntersectingLine where
  m : ℝ  -- slope of the line
  q : ℝ  -- y-intercept of the line

/-- The intersection points of tangents to a hyperbola -/
structure TangentIntersection where
  x : ℝ
  y : ℝ

/-- Calculate the intersection point of tangents for y = 1/x hyperbola -/
noncomputable def tangentIntersection1 (line : IntersectingLine) : TangentIntersection :=
  { x := 2 * line.m / line.q
  , y := -2 / line.q }

/-- Calculate the intersection point of tangents for y = -1/x hyperbola -/
noncomputable def tangentIntersection2 (line : IntersectingLine) : TangentIntersection :=
  { x := -2 * line.m / line.q
  , y := 2 / line.q }

/-- Two points are symmetric about the origin -/
def symmetricAboutOrigin (p1 p2 : TangentIntersection) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- Main theorem: The intersection points of tangents are symmetric about the origin -/
theorem tangent_intersections_symmetric (line : IntersectingLine) :
  symmetricAboutOrigin (tangentIntersection1 line) (tangentIntersection2 line) := by
  sorry

end tangent_intersections_symmetric_l1455_145522


namespace complex_calculation_l1455_145523

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Define w as a function of z
def w (z : ℂ) : ℂ := z^2 + 3 - 4

-- Theorem statement
theorem complex_calculation :
  w z = 2 * Complex.I - 1 := by sorry

end complex_calculation_l1455_145523


namespace sin_negative_thirty_degrees_l1455_145507

theorem sin_negative_thirty_degrees : 
  Real.sin (-(30 * π / 180)) = -(1 / 2) := by
  sorry

end sin_negative_thirty_degrees_l1455_145507


namespace quadratic_sum_l1455_145594

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ), 
  (15 * x^2 + 150 * x + 2250 = a * (x + b)^2 + c) ∧ (a + b + c = 1895) := by
  sorry

end quadratic_sum_l1455_145594


namespace functional_equation_equivalence_l1455_145557

theorem functional_equation_equivalence (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x + f y) ↔ (∀ x y, f (x + y + x * y) = f x + f y + f (x * y)) :=
sorry

end functional_equation_equivalence_l1455_145557


namespace x_intercept_of_line_l1455_145559

/-- The x-intercept of the line 4x + 7y = 28 is the point (7, 0). -/
theorem x_intercept_of_line (x y : ℝ) : 
  (4 * x + 7 * y = 28) → (x = 7 ∧ y = 0 → 4 * x + 7 * y = 28) := by
  sorry

end x_intercept_of_line_l1455_145559


namespace youngest_sibling_age_l1455_145512

/-- Given 4 siblings where 3 are 4, 5, and 7 years older than the youngest,
    and their average age is 21, prove that the age of the youngest sibling is 17. -/
theorem youngest_sibling_age (y : ℕ) : 
  (y + (y + 4) + (y + 5) + (y + 7)) / 4 = 21 → y = 17 := by
  sorry

end youngest_sibling_age_l1455_145512


namespace number_of_divisors_2001_l1455_145579

theorem number_of_divisors_2001 : Finset.card (Nat.divisors 2001) = 8 := by
  sorry

end number_of_divisors_2001_l1455_145579


namespace complex_magnitude_l1455_145592

theorem complex_magnitude (a b : ℝ) (z : ℂ) :
  (a + Complex.I)^2 = b * Complex.I →
  z = a + b * Complex.I →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l1455_145592


namespace quadratic_inequality_theorem_l1455_145558

-- Define the quadratic inequality and its solution set
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop := (1 - a) * x^2 - 4*x + 6 > 0
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}

-- State the theorem
theorem quadratic_inequality_theorem (a : ℝ) :
  (∀ x, x ∈ solution_set a ↔ quadratic_inequality a x) →
  (a = 3) ∧
  (∀ x, 2*x^2 + (2-a)*x - a > 0 ↔ x < -1 ∨ x > 1) ∧
  (∀ b, (∀ x, a*x^2 + b*x + 3 ≥ 0) ↔ -6 ≤ b ∧ b ≤ 6) :=
sorry

end quadratic_inequality_theorem_l1455_145558


namespace division_remainder_l1455_145510

theorem division_remainder (N : ℕ) : 
  (∃ r : ℕ, N = 5 * 5 + r ∧ r < 5) ∧ 
  (∃ q : ℕ, N = 11 * q + 3) → 
  N % 5 = 0 := by sorry

end division_remainder_l1455_145510


namespace intersection_implies_sum_l1455_145577

/-- Given two functions f and g defined as:
    f(x) = -2|x-a| + b
    g(x) = 2|x-c| + d
    If f(5) = g(5) = 10 and f(11) = g(11) = 6,
    then a + c = 16 -/
theorem intersection_implies_sum (a b c d : ℝ) :
  (∀ x, -2 * |x - a| + b = 2 * |x - c| + d → x = 5 ∨ x = 11) →
  -2 * |5 - a| + b = 10 →
  2 * |5 - c| + d = 10 →
  -2 * |11 - a| + b = 6 →
  2 * |11 - c| + d = 6 →
  a + c = 16 := by
  sorry

end intersection_implies_sum_l1455_145577


namespace min_max_sum_l1455_145534

theorem min_max_sum (x y z u v : ℕ+) (h : x + y + z + u + v = 2505) :
  let N := max (x + y) (max (y + z) (max (z + u) (u + v)))
  N ≥ 1253 ∧ ∃ (a b c d e : ℕ+), a + b + c + d + e = 2505 ∧
    max (a + b) (max (b + c) (max (c + d) (d + e))) = 1253 := by
  sorry

end min_max_sum_l1455_145534
