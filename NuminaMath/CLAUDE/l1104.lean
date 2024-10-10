import Mathlib

namespace max_product_of_three_integers_l1104_110401

theorem max_product_of_three_integers (a b c : ℕ+) : 
  (a * b * c = 8 * (a + b + c)) → (c = a + b) → (a * b * c ≤ 272) := by
  sorry

end max_product_of_three_integers_l1104_110401


namespace parallel_vectors_x_value_l1104_110463

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
sorry

end parallel_vectors_x_value_l1104_110463


namespace boat_speed_in_still_water_l1104_110489

/-- The speed of a boat in still water, given its travel times with varying current and wind conditions. -/
theorem boat_speed_in_still_water 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (current_start : ℝ) 
  (current_end : ℝ) 
  (wind_slowdown : ℝ) 
  (h1 : downstream_time = 3)
  (h2 : upstream_time = 4.5)
  (h3 : current_start = 2)
  (h4 : current_end = 4)
  (h5 : wind_slowdown = 1) :
  ∃ (boat_speed : ℝ), boat_speed = 16 ∧ 
  (boat_speed + (current_start + current_end) / 2 - wind_slowdown) * downstream_time = 
  (boat_speed - (current_start + current_end) / 2 - wind_slowdown) * upstream_time :=
by sorry

end boat_speed_in_still_water_l1104_110489


namespace surface_area_difference_specific_l1104_110498

/-- Calculates the surface area difference when removing a cube from a rectangular solid -/
def surface_area_difference (l w h : ℝ) (cube_side : ℝ) : ℝ :=
  let original_surface_area := 2 * (l * w + l * h + w * h)
  let new_faces_area := 2 * cube_side * cube_side
  let removed_faces_area := 5 * cube_side * cube_side
  new_faces_area - removed_faces_area

/-- The surface area difference for the specific problem -/
theorem surface_area_difference_specific :
  surface_area_difference 6 5 4 2 = -12 := by
  sorry

end surface_area_difference_specific_l1104_110498


namespace smartphone_price_difference_l1104_110418

def store_a_full_price : ℚ := 125
def store_a_discount : ℚ := 8 / 100
def store_b_full_price : ℚ := 130
def store_b_discount : ℚ := 10 / 100

theorem smartphone_price_difference :
  store_b_full_price * (1 - store_b_discount) - store_a_full_price * (1 - store_a_discount) = 2 := by
  sorry

end smartphone_price_difference_l1104_110418


namespace three_a_in_S_implies_a_in_S_l1104_110470

theorem three_a_in_S_implies_a_in_S (a : ℤ) : 
  (∃ x y : ℤ, 3 * a = x^2 + 2 * y^2) → 
  (∃ u v : ℤ, a = u^2 + 2 * v^2) := by
sorry

end three_a_in_S_implies_a_in_S_l1104_110470


namespace simplify_expression_1_simplify_expression_2_l1104_110403

-- Problem 1
theorem simplify_expression_1 (x y z : ℝ) :
  (x + y + z)^2 - (x + y - z)^2 = 4*z*(x + y) := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  (a + 2*b)^2 - 2*(a + 2*b)*(a - 2*b) + (a - 2*b)^2 = 16*b^2 := by sorry

end simplify_expression_1_simplify_expression_2_l1104_110403


namespace arithmetic_sequence_ninth_term_l1104_110421

/-- Given an arithmetic sequence where the third term is 23 and the sixth term is 29,
    prove that the ninth term is 35. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℕ)  -- a is the sequence
  (h1 : a 3 = 23)  -- third term is 23
  (h2 : a 6 = 29)  -- sixth term is 29
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence property
  : a 9 = 35 := by
  sorry

end arithmetic_sequence_ninth_term_l1104_110421


namespace largest_valid_n_l1104_110488

/-- Represents the color of a ball -/
inductive Color
| Black
| White

/-- Represents a coloring function for balls -/
def Coloring := ℕ → Color

/-- Checks if a coloring satisfies the given condition -/
def ValidColoring (c : Coloring) (n : ℕ) : Prop :=
  ∀ a₁ a₂ a₃ a₄ : ℕ,
    a₁ ≤ n ∧ a₂ ≤ n ∧ a₃ ≤ n ∧ a₄ ≤ n →
    a₁ + a₂ + a₃ = a₄ →
    (c a₁ = Color.Black ∨ c a₂ = Color.Black ∨ c a₃ = Color.Black) ∧
    (c a₁ = Color.White ∨ c a₂ = Color.White ∨ c a₃ = Color.White)

/-- The theorem stating that 10 is the largest possible value of n -/
theorem largest_valid_n :
  (∃ c : Coloring, ValidColoring c 10) ∧
  (∀ n > 10, ¬∃ c : Coloring, ValidColoring c n) :=
sorry

end largest_valid_n_l1104_110488


namespace impossibleArrangement_l1104_110456

/-- Represents a person at the table -/
structure Person :=
  (index : Fin 40)

/-- Represents the circular table with 40 people -/
def Table := Fin 40 → Person

/-- Calculates the number of people between two given people -/
def distance (table : Table) (p1 p2 : Person) : Nat :=
  sorry

/-- Determines if two people have a mutual acquaintance -/
def hasCommonAcquaintance (table : Table) (p1 p2 : Person) : Prop :=
  sorry

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossibleArrangement :
  ¬ ∃ (table : Table),
    (∀ (p1 p2 : Person),
      hasCommonAcquaintance table p1 p2 ↔ Even (distance table p1 p2)) :=
  sorry

end impossibleArrangement_l1104_110456


namespace divisible_by_thirteen_l1104_110408

theorem divisible_by_thirteen (a b : ℕ) (h : a * 13 = 119268916) :
  119268903 % 13 = 0 := by
  sorry

end divisible_by_thirteen_l1104_110408


namespace train_route_encoding_l1104_110499

def encode_letter (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

def decode_digit (n : ℕ) : Char :=
  Char.ofNat (n + 'A'.toNat - 1)

def encode_city (s : String) : List ℕ :=
  s.toList.map encode_letter

theorem train_route_encoding :
  (encode_city "UFA" = [21, 6, 1]) ∧
  (encode_city "BAKU" = [2, 1, 11, 21]) →
  "21221-211221".splitOn "-" = ["21221", "211221"] →
  ∃ (departure arrival : String),
    departure = "UFA" ∧
    arrival = "BAKU" ∧
    encode_city departure = [21, 6, 1] ∧
    encode_city arrival = [2, 1, 11, 21] :=
by sorry

end train_route_encoding_l1104_110499


namespace logarithm_sum_property_l1104_110436

theorem logarithm_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by sorry

end logarithm_sum_property_l1104_110436


namespace point_not_on_transformed_plane_l1104_110414

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane3D) (k : ℝ) : Plane3D :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem -/
theorem point_not_on_transformed_plane :
  let A : Point3D := { x := 5, y := 0, z := -1 }
  let a : Plane3D := { a := 2, b := -1, c := 3, d := -1 }
  let k : ℝ := 3
  let a' : Plane3D := transformPlane a k
  ¬ pointOnPlane A a' := by
  sorry

end point_not_on_transformed_plane_l1104_110414


namespace unique_number_l1104_110490

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  n % 2 = 1 ∧ 
  n % 13 = 0 ∧ 
  is_perfect_square (digit_product n) ∧
  n = 91 := by
sorry

end unique_number_l1104_110490


namespace sqrt_D_irrational_l1104_110434

theorem sqrt_D_irrational (k : ℤ) : 
  let a : ℤ := 3 * k
  let b : ℤ := 3 * k + 3
  let c : ℤ := a + b
  let D : ℤ := a^2 + b^2 + c^2
  Irrational (Real.sqrt D) := by sorry

end sqrt_D_irrational_l1104_110434


namespace total_profit_is_56700_l1104_110467

/-- Given a profit sharing ratio and c's profit, calculate the total profit -/
def calculate_total_profit (ratio_a ratio_b ratio_c : ℕ) (profit_c : ℕ) : ℕ :=
  let total_parts := ratio_a + ratio_b + ratio_c
  let part_value := profit_c / ratio_c
  total_parts * part_value

/-- Theorem: The total profit is $56,700 given the specified conditions -/
theorem total_profit_is_56700 :
  calculate_total_profit 8 9 10 21000 = 56700 := by
  sorry

end total_profit_is_56700_l1104_110467


namespace discount_calculation_l1104_110483

/-- Given an article with a cost price of 100 units, if the selling price is marked 12% above 
    the cost price and the trader suffers a loss of 1% at the time of selling, 
    then the discount allowed is 13 units. -/
theorem discount_calculation (cost_price : ℝ) (marked_price : ℝ) (selling_price : ℝ) : 
  cost_price = 100 →
  marked_price = cost_price * 1.12 →
  selling_price = cost_price * 0.99 →
  marked_price - selling_price = 13 :=
by sorry

end discount_calculation_l1104_110483


namespace prob_multiple_of_3_twice_in_four_rolls_l1104_110430

/-- The probability of rolling a multiple of 3 on a fair six-sided die -/
def prob_multiple_of_3 : ℚ := 1 / 3

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 4

/-- The number of times we want to see a multiple of 3 -/
def target_occurrences : ℕ := 2

/-- The probability of rolling a multiple of 3 exactly twice in four rolls of a fair die -/
theorem prob_multiple_of_3_twice_in_four_rolls :
  Nat.choose num_rolls target_occurrences * prob_multiple_of_3 ^ target_occurrences * (1 - prob_multiple_of_3) ^ (num_rolls - target_occurrences) = 8 / 27 := by
  sorry

end prob_multiple_of_3_twice_in_four_rolls_l1104_110430


namespace prob_at_least_one_multiple_of_four_l1104_110440

def probability_at_least_one_multiple_of_four : ℚ :=
  let total_numbers : ℕ := 60
  let multiples_of_four : ℕ := 15
  let prob_not_multiple : ℚ := (total_numbers - multiples_of_four) / total_numbers
  1 - prob_not_multiple ^ 2

theorem prob_at_least_one_multiple_of_four :
  probability_at_least_one_multiple_of_four = 7 / 16 := by
  sorry

end prob_at_least_one_multiple_of_four_l1104_110440


namespace diagonals_30_sided_polygon_l1104_110496

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

end diagonals_30_sided_polygon_l1104_110496


namespace sum_of_numbers_l1104_110495

-- Define the range of numbers
def valid_number (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 50

-- Define primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define Alice's number
def alice_number (a : ℕ) : Prop := valid_number a

-- Define Bob's number
def bob_number (b : ℕ) : Prop := valid_number b ∧ is_prime b

-- Alice can't determine who has the larger number
def alice_uncertainty (a b : ℕ) : Prop :=
  alice_number a → bob_number b → ¬(a > b ∨ b > a)

-- Bob can determine who has the larger number after Alice's statement
def bob_certainty (a b : ℕ) : Prop :=
  alice_number a → bob_number b → (a > b ∨ b > a)

-- 200 * Bob's number + Alice's number is a perfect square
def perfect_square_condition (a b : ℕ) : Prop :=
  alice_number a → bob_number b → ∃ k : ℕ, 200 * b + a = k * k

-- Theorem statement
theorem sum_of_numbers (a b : ℕ) :
  alice_number a →
  bob_number b →
  alice_uncertainty a b →
  bob_certainty a b →
  perfect_square_condition a b →
  a + b = 43 := by
  sorry

end sum_of_numbers_l1104_110495


namespace marys_stickers_l1104_110458

theorem marys_stickers (total_stickers : ℕ) (friends : ℕ) (other_students : ℕ) (stickers_per_other : ℕ) (leftover_stickers : ℕ) (total_students : ℕ) :
  total_stickers = 50 →
  friends = 5 →
  other_students = total_students - friends - 1 →
  stickers_per_other = 2 →
  leftover_stickers = 8 →
  total_students = 17 →
  (total_stickers - leftover_stickers - other_students * stickers_per_other) / friends = 4 := by
  sorry

#check marys_stickers

end marys_stickers_l1104_110458


namespace divisibility_by_five_l1104_110478

theorem divisibility_by_five (a b : ℕ) : 
  (5 ∣ (a * b)) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end divisibility_by_five_l1104_110478


namespace smallest_integer_multiple_conditions_l1104_110448

theorem smallest_integer_multiple_conditions :
  ∃ n : ℕ, n > 0 ∧
  (∃ k : ℤ, n = 5 * k + 3) ∧
  (∃ m : ℤ, n = 12 * m) ∧
  (∀ x : ℕ, x > 0 →
    (∃ k' : ℤ, x = 5 * k' + 3) →
    (∃ m' : ℤ, x = 12 * m') →
    n ≤ x) ∧
  n = 48 :=
by sorry

end smallest_integer_multiple_conditions_l1104_110448


namespace initial_men_count_l1104_110469

/-- Given a group of men with provisions lasting 18 days, prove that the initial number of men is 1000 
    when 400 more men join and the provisions then last 12.86 days, assuming the total amount of provisions remains constant. -/
theorem initial_men_count (initial_days : ℝ) (final_days : ℝ) (additional_men : ℕ) :
  initial_days = 18 →
  final_days = 12.86 →
  additional_men = 400 →
  ∃ (initial_men : ℕ), 
    initial_men * initial_days = (initial_men + additional_men) * final_days ∧
    initial_men = 1000 := by
  sorry

end initial_men_count_l1104_110469


namespace quadratic_equation_solution_l1104_110480

theorem quadratic_equation_solution : ∃ x : ℝ, (10 - x)^2 = x^2 + 6 ∧ x = 4.7 := by
  sorry

end quadratic_equation_solution_l1104_110480


namespace complex_magnitude_product_l1104_110466

theorem complex_magnitude_product : Complex.abs (3 - 5 * Complex.I) * Complex.abs (3 + 5 * Complex.I) = 34 := by
  sorry

end complex_magnitude_product_l1104_110466


namespace tom_car_washing_earnings_l1104_110494

/-- 
Given:
- Tom had $74 last week
- Tom has $160 now
Prove that Tom made $86 by washing cars over the weekend.
-/
theorem tom_car_washing_earnings :
  let initial_money : ℕ := 74
  let current_money : ℕ := 160
  let money_earned : ℕ := current_money - initial_money
  money_earned = 86 := by sorry

end tom_car_washing_earnings_l1104_110494


namespace cube_root_equation_solution_l1104_110442

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - x / 3) ^ (1/3 : ℝ) = -2 :=
by sorry

end cube_root_equation_solution_l1104_110442


namespace mary_baseball_cards_l1104_110438

def baseball_cards (initial cards_from_fred cards_bought torn : ℕ) : ℕ :=
  initial - torn + cards_from_fred + cards_bought

theorem mary_baseball_cards : 
  baseball_cards 18 26 40 8 = 76 := by
  sorry

end mary_baseball_cards_l1104_110438


namespace remainder_sum_mod_59_l1104_110479

theorem remainder_sum_mod_59 (a b c : ℕ+) 
  (ha : a ≡ 28 [ZMOD 59])
  (hb : b ≡ 34 [ZMOD 59])
  (hc : c ≡ 5 [ZMOD 59]) :
  (a + b + c) ≡ 8 [ZMOD 59] := by
  sorry

end remainder_sum_mod_59_l1104_110479


namespace cubic_sum_theorem_l1104_110449

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_cond : a + b + c = 2)
  (prod_sum_cond : a * b + a * c + b * c = -3)
  (prod_cond : a * b * c = -3) :
  a^3 + b^3 + c^3 = 6 := by
  sorry

end cubic_sum_theorem_l1104_110449


namespace min_socks_for_15_pairs_l1104_110464

/-- Represents the number of socks of each color in the drawer -/
def Drawer := List Nat

/-- The total number of socks in the drawer -/
def total_socks (d : Drawer) : Nat :=
  d.sum

/-- The number of different colors of socks in the drawer -/
def num_colors (d : Drawer) : Nat :=
  d.length

/-- The minimum number of socks needed to guarantee a certain number of pairs -/
def min_socks_for_pairs (num_pairs : Nat) (num_colors : Nat) : Nat :=
  num_colors + 2 * (num_pairs - 1)

theorem min_socks_for_15_pairs (d : Drawer) :
  num_colors d = 5 →
  total_socks d ≥ 400 →
  min_socks_for_pairs 15 (num_colors d) = 33 :=
by sorry

end min_socks_for_15_pairs_l1104_110464


namespace planes_perpendicular_from_line_perpendicular_planes_from_parallel_l1104_110481

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem 1
theorem planes_perpendicular_from_line 
  (α β : Plane) (l : Line) :
  line_perpendicular l α → line_parallel l β → perpendicular α β := by sorry

-- Theorem 2
theorem perpendicular_planes_from_parallel 
  (α β γ : Plane) :
  parallel α β → perpendicular α γ → perpendicular β γ := by sorry

end planes_perpendicular_from_line_perpendicular_planes_from_parallel_l1104_110481


namespace lines_perp_to_plane_are_parallel_l1104_110459

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel 
  (a b : Line) (α : Plane) 
  (h1 : perp a α) (h2 : perp b α) : 
  para a b := by sorry

end lines_perp_to_plane_are_parallel_l1104_110459


namespace janice_working_days_l1104_110491

-- Define the problem parameters
def regular_pay : ℕ := 30
def overtime_pay : ℕ := 15
def overtime_shifts : ℕ := 3
def total_earnings : ℕ := 195

-- Define the function to calculate the number of working days
def calculate_working_days (regular_pay overtime_pay overtime_shifts total_earnings : ℕ) : ℕ :=
  (total_earnings - overtime_pay * overtime_shifts) / regular_pay

-- Theorem statement
theorem janice_working_days :
  calculate_working_days regular_pay overtime_pay overtime_shifts total_earnings = 5 := by
  sorry


end janice_working_days_l1104_110491


namespace courtyard_width_l1104_110465

/-- The width of a rectangular courtyard given its length and paving stone requirements. -/
theorem courtyard_width
  (length : ℝ)
  (num_stones : ℕ)
  (stone_length : ℝ)
  (stone_width : ℝ)
  (h1 : length = 50)
  (h2 : num_stones = 165)
  (h3 : stone_length = 5/2)
  (h4 : stone_width = 2)
  : (num_stones * stone_length * stone_width) / length = 33/2 :=
by sorry

end courtyard_width_l1104_110465


namespace lifeguard_swimming_distance_l1104_110411

/-- The problem of calculating the total swimming distance for a lifeguard test. -/
theorem lifeguard_swimming_distance 
  (front_crawl_speed : ℝ) 
  (breaststroke_speed : ℝ) 
  (total_time : ℝ) 
  (front_crawl_time : ℝ) 
  (h1 : front_crawl_speed = 45) 
  (h2 : breaststroke_speed = 35) 
  (h3 : total_time = 12) 
  (h4 : front_crawl_time = 8) :
  front_crawl_speed * front_crawl_time + breaststroke_speed * (total_time - front_crawl_time) = 500 := by
  sorry

#check lifeguard_swimming_distance

end lifeguard_swimming_distance_l1104_110411


namespace quadratic_equation_solution_algebraic_simplification_l1104_110431

-- Part 1: Quadratic equation
theorem quadratic_equation_solution (x : ℝ) : 
  2 * x^2 - 3 * x + 1 = 0 ↔ x = 1/2 ∨ x = 1 := by sorry

-- Part 2: Algebraic simplification
theorem algebraic_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  ((a^2 - b^2) / (a^2 - 2*a*b + b^2) + a / (b - a)) / (b^2 / (a^2 - a*b)) = a / b := by sorry

end quadratic_equation_solution_algebraic_simplification_l1104_110431


namespace crosswalk_stripe_distance_l1104_110433

/-- Given a street with parallel curbs and a crosswalk, calculate the distance between the stripes -/
theorem crosswalk_stripe_distance
  (curb_distance : ℝ)
  (curb_length : ℝ)
  (stripe_length : ℝ)
  (h_curb_distance : curb_distance = 50)
  (h_curb_length : curb_length = 20)
  (h_stripe_length : stripe_length = 65) :
  (curb_distance * curb_length) / stripe_length = 200 / 13 := by
sorry

end crosswalk_stripe_distance_l1104_110433


namespace soccer_league_female_fraction_l1104_110484

/-- Represents the number of participants in a soccer league for two consecutive years -/
structure LeagueParticipation where
  malesLastYear : ℕ
  femalesLastYear : ℕ
  malesThisYear : ℕ
  femalesThisYear : ℕ

/-- Calculates the fraction of female participants this year -/
def femaleFraction (lp : LeagueParticipation) : Rat :=
  lp.femalesThisYear / (lp.malesThisYear + lp.femalesThisYear)

theorem soccer_league_female_fraction 
  (lp : LeagueParticipation)
  (male_increase : lp.malesThisYear = (110 * lp.malesLastYear) / 100)
  (female_increase : lp.femalesThisYear = (125 * lp.femalesLastYear) / 100)
  (total_increase : lp.malesThisYear + lp.femalesThisYear = 
    (115 * (lp.malesLastYear + lp.femalesLastYear)) / 100)
  (males_last_year : lp.malesLastYear = 30)
  : femaleFraction lp = 19 / 52 := by
  sorry

#check soccer_league_female_fraction

end soccer_league_female_fraction_l1104_110484


namespace solve_for_x_l1104_110420

theorem solve_for_x (M N : ℝ) (h1 : M = 2*x - 4) (h2 : N = 2*x + 3) (h3 : 3*M - N = 1) : x = 4 := by
  sorry

end solve_for_x_l1104_110420


namespace book_distribution_theorem_l1104_110460

/-- Represents the number of ways to distribute books between the library and checked-out status. -/
def book_distribution_ways (n₁ n₂ : ℕ) : ℕ :=
  (n₁ - 1) * (n₂ - 1)

/-- Theorem stating the number of ways to distribute books between the library and checked-out status. -/
theorem book_distribution_theorem :
  let n₁ : ℕ := 8  -- number of copies of the first type of book
  let n₂ : ℕ := 4  -- number of copies of the second type of book
  book_distribution_ways n₁ n₂ = 21 :=
by
  sorry

#eval book_distribution_ways 8 4

end book_distribution_theorem_l1104_110460


namespace inequality_proof_l1104_110419

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (2 * a^2 + 3 * b^2 ≥ 6/5) ∧ ((a + 1/a) * (b + 1/b) ≥ 25/4) := by
  sorry

end inequality_proof_l1104_110419


namespace intersection_is_empty_l1104_110425

def A : Set ℝ := {α | ∃ k : ℤ, α = (5 * k * Real.pi) / 3}
def B : Set ℝ := {β | ∃ k : ℤ, β = (3 * k * Real.pi) / 2}

theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end intersection_is_empty_l1104_110425


namespace intersection_A_complement_B_range_of_a_l1104_110416

-- Define the sets A, B, and M
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | x * (3 - x) > 0}
def M (a : ℝ) : Set ℝ := {x | 2 * x - a < 0}

-- Theorem for part 1
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x ≤ 0} := by sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) : (A ∪ B) ⊆ M a → a ≥ 6 := by sorry

end intersection_A_complement_B_range_of_a_l1104_110416


namespace mr_a_net_gain_l1104_110415

/-- The total net gain for Mr. A after a series of house transactions -/
theorem mr_a_net_gain (house1_value house2_value : ℝ)
  (house1_profit house1_loss house2_profit house2_loss : ℝ)
  (h1 : house1_value = 15000)
  (h2 : house2_value = 20000)
  (h3 : house1_profit = 0.15)
  (h4 : house1_loss = 0.15)
  (h5 : house2_profit = 0.20)
  (h6 : house2_loss = 0.20) :
  let sale1 := house1_value * (1 + house1_profit)
  let sale2 := house2_value * (1 + house2_profit)
  let buyback1 := sale1 * (1 - house1_loss)
  let buyback2 := sale2 * (1 - house2_loss)
  let net_gain := (sale1 - buyback1) + (sale2 - buyback2)
  net_gain = 7387.50 := by
  sorry

end mr_a_net_gain_l1104_110415


namespace cubic_equation_root_l1104_110410

theorem cubic_equation_root (a b : ℚ) : 
  (3 - 5 * Real.sqrt 2)^3 + a * (3 - 5 * Real.sqrt 2)^2 + b * (3 - 5 * Real.sqrt 2) - 47 = 0 → 
  a = -199/41 := by
sorry

end cubic_equation_root_l1104_110410


namespace opposite_of_seven_l1104_110437

theorem opposite_of_seven : 
  (-(7 : ℝ) = -7) := by sorry

end opposite_of_seven_l1104_110437


namespace similar_triangles_side_ratio_l1104_110450

theorem similar_triangles_side_ratio 
  (a b ka kb : ℝ) 
  (C : Real) 
  (k : ℝ) 
  (h1 : ka = k * a) 
  (h2 : kb = k * b) 
  (h3 : C > 0 ∧ C < 180) : 
  ∃ (c kc : ℝ), c > 0 ∧ kc > 0 ∧ kc = k * c :=
sorry

end similar_triangles_side_ratio_l1104_110450


namespace matthew_friends_count_l1104_110402

def total_crackers : ℝ := 36
def crackers_per_friend : ℝ := 6.5

theorem matthew_friends_count :
  ⌊total_crackers / crackers_per_friend⌋ = 5 :=
by sorry

end matthew_friends_count_l1104_110402


namespace probability_same_parity_l1104_110492

-- Define the type for function parity
inductive Parity
| Even
| Odd
| Neither

-- Define a function to represent the parity of each given function
def function_parity : Fin 4 → Parity
| 0 => Parity.Neither  -- y = x^3 + 3x^2
| 1 => Parity.Even     -- y = (e^x + e^-x) / 2
| 2 => Parity.Odd      -- y = log_2 ((3-x)/(3+x))
| 3 => Parity.Even     -- y = x sin x

-- Define a function to check if two functions have the same parity
def same_parity (f1 f2 : Fin 4) : Bool :=
  match function_parity f1, function_parity f2 with
  | Parity.Even, Parity.Even => true
  | Parity.Odd, Parity.Odd => true
  | _, _ => false

-- Theorem statement
theorem probability_same_parity :
  (Finset.filter (fun p => same_parity p.1 p.2) (Finset.univ : Finset (Fin 4 × Fin 4))).card /
  (Finset.univ : Finset (Fin 4 × Fin 4)).card = 1 / 6 :=
sorry

end probability_same_parity_l1104_110492


namespace janettes_remaining_jerky_l1104_110473

def camping_days : ℕ := 5
def initial_jerky : ℕ := 40
def breakfast_jerky : ℕ := 1
def lunch_jerky : ℕ := 1
def dinner_jerky : ℕ := 2

def daily_consumption : ℕ := breakfast_jerky + lunch_jerky + dinner_jerky

def total_consumed : ℕ := daily_consumption * camping_days

def remaining_after_trip : ℕ := initial_jerky - total_consumed

def given_to_brother : ℕ := remaining_after_trip / 2

theorem janettes_remaining_jerky :
  initial_jerky - total_consumed - given_to_brother = 10 := by
  sorry

end janettes_remaining_jerky_l1104_110473


namespace smallest_factor_for_perfect_square_l1104_110487

theorem smallest_factor_for_perfect_square : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (m : ℕ), 1152 * n = m^2) ∧ 
  (∀ (k : ℕ), k > 0 → k < n → ¬∃ (m : ℕ), 1152 * k = m^2) ∧
  n = 6 :=
by sorry

end smallest_factor_for_perfect_square_l1104_110487


namespace sweet_shop_candy_cases_l1104_110412

/-- The number of cases of chocolate bars in the Sweet Shop -/
def chocolate_cases : ℕ := 80 - 55

/-- The total number of cases of candy in the Sweet Shop -/
def total_cases : ℕ := 80

/-- The number of cases of lollipops in the Sweet Shop -/
def lollipop_cases : ℕ := 55

theorem sweet_shop_candy_cases : chocolate_cases = 25 := by
  sorry

end sweet_shop_candy_cases_l1104_110412


namespace zoo_elephant_count_l1104_110453

/-- Represents the number of animals of each type in the zoo -/
structure ZooPopulation where
  giraffes : ℕ
  penguins : ℕ
  elephants : ℕ
  total : ℕ

/-- The conditions of the zoo population -/
def zoo_conditions (pop : ZooPopulation) : Prop :=
  pop.giraffes = 5 ∧
  pop.penguins = 2 * pop.giraffes ∧
  pop.penguins = (20 : ℕ) * pop.total / 100 ∧
  pop.elephants = (4 : ℕ) * pop.total / 100 ∧
  pop.total = pop.giraffes + pop.penguins + pop.elephants

theorem zoo_elephant_count :
  ∀ pop : ZooPopulation, zoo_conditions pop → pop.elephants = 2 :=
by sorry

end zoo_elephant_count_l1104_110453


namespace quadratic_transformation_l1104_110441

def f (x : ℝ) : ℝ := x^2

def shift_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x - a)

def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x => f x + b

def g (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem quadratic_transformation :
  shift_up (shift_right f 3) 4 = g := by sorry

end quadratic_transformation_l1104_110441


namespace f_increasing_on_interval_l1104_110451

def f (x : ℝ) : ℝ := 3 * x^2 + 8 * x - 10

theorem f_increasing_on_interval : 
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y := by
  sorry

end f_increasing_on_interval_l1104_110451


namespace range_of_x_l1104_110477

theorem range_of_x (x : ℝ) : 
  (¬ (x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)) → (1 ≤ x ∧ x < 2) :=
by sorry

end range_of_x_l1104_110477


namespace thabos_books_l1104_110427

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def validCollection (books : BookCollection) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 160 ∧
  books.paperbackNonfiction > books.hardcoverNonfiction ∧
  books.paperbackFiction = 2 * books.paperbackNonfiction ∧
  books.hardcoverNonfiction = 25

theorem thabos_books (books : BookCollection) (h : validCollection books) :
  books.paperbackNonfiction - books.hardcoverNonfiction = 20 := by
  sorry

end thabos_books_l1104_110427


namespace total_like_count_l1104_110409

/-- Represents the number of employees with a "dislike" attitude -/
def dislike_count : ℕ := sorry

/-- Represents the number of employees with a "neutral" attitude -/
def neutral_count : ℕ := dislike_count + 12

/-- Represents the number of employees with a "like" attitude -/
def like_count : ℕ := 6 * dislike_count

/-- Represents the ratio of employees with each attitude in the stratified sample -/
def sample_ratio : ℕ × ℕ × ℕ := (6, 1, 3)

theorem total_like_count : like_count = 36 := by sorry

end total_like_count_l1104_110409


namespace pure_imaginary_condition_l1104_110497

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * (a - Complex.I * (a + 2)) = 0) → a = -2 := by
  sorry

end pure_imaginary_condition_l1104_110497


namespace loss_percentage_proof_l1104_110454

def cost_price : ℝ := 1250
def price_increase : ℝ := 500
def gain_percentage : ℝ := 0.15

theorem loss_percentage_proof (selling_price : ℝ) 
  (h1 : selling_price + price_increase = cost_price * (1 + gain_percentage)) :
  (cost_price - selling_price) / cost_price = 0.25 := by
  sorry

end loss_percentage_proof_l1104_110454


namespace zoe_mp3_songs_l1104_110475

theorem zoe_mp3_songs (initial_songs : ℕ) (deleted_songs : ℕ) (added_songs : ℕ) :
  initial_songs = 6 →
  deleted_songs = 3 →
  added_songs = 20 →
  initial_songs - deleted_songs + added_songs = 23 :=
by sorry

end zoe_mp3_songs_l1104_110475


namespace factorization_equality_l1104_110486

theorem factorization_equality (a b : ℝ) : 2 * a^3 - 8 * a * b^2 = 2 * a * (a + 2*b) * (a - 2*b) := by
  sorry

end factorization_equality_l1104_110486


namespace jump_difference_l1104_110457

def monday_jumps : ℕ := 88
def tuesday_jumps : ℕ := 75
def wednesday_jumps : ℕ := 62
def thursday_jumps : ℕ := 91
def friday_jumps : ℕ := 80

def jump_counts : List ℕ := [monday_jumps, tuesday_jumps, wednesday_jumps, thursday_jumps, friday_jumps]

theorem jump_difference :
  (List.maximum jump_counts).get! - (List.minimum jump_counts).get! = 29 := by
  sorry

end jump_difference_l1104_110457


namespace angle_sum_in_square_configuration_l1104_110429

/-- Given a configuration of 13 identical squares with marked points, this theorem proves
    that the sum of specific angles equals 405 degrees. -/
theorem angle_sum_in_square_configuration :
  ∀ (FPB FPD APC APE AQG QCF RQF CQD : ℝ),
  RQF + CQD = 45 →
  FPB + FPD + APE = 180 →
  AQG + QCF + APC = 180 →
  (FPB + FPD + APC + APE) + (AQG + QCF + RQF + CQD) = 405 := by
  sorry

end angle_sum_in_square_configuration_l1104_110429


namespace function_roots_l1104_110445

def has_at_least_roots (f : ℝ → ℝ) (n : ℕ) (a b : ℝ) : Prop :=
  ∃ (S : Finset ℝ), S.card ≥ n ∧ (∀ x ∈ S, a ≤ x ∧ x ≤ b ∧ f x = 0)

theorem function_roots (g : ℝ → ℝ) 
  (h1 : ∀ x, g (3 + x) = g (3 - x))
  (h2 : ∀ x, g (8 + x) = g (8 - x))
  (h3 : g 0 = 0) :
  has_at_least_roots g 501 (-2000) 2000 := by
  sorry

end function_roots_l1104_110445


namespace inequality_range_of_a_l1104_110413

theorem inequality_range_of_a (a : ℝ) : 
  (∀ x > 0, Real.log x + a * x + 1 - x * Real.exp (2 * x) ≤ 0) → 
  a ≤ 2 := by
sorry

end inequality_range_of_a_l1104_110413


namespace negation_of_quadratic_inequality_l1104_110446

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 + 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 1 ≤ 0) := by
  sorry

end negation_of_quadratic_inequality_l1104_110446


namespace equation_solution_l1104_110406

theorem equation_solution :
  let f (x : ℚ) := (3 - x) / (x + 2) + (3*x - 6) / (3 - x)
  ∃! x, f x = 2 ∧ x = -7/6 :=
by
  sorry

end equation_solution_l1104_110406


namespace literary_readers_count_l1104_110439

theorem literary_readers_count (total : ℕ) (sci_fi : ℕ) (both : ℕ) (lit : ℕ) :
  total = 400 →
  sci_fi = 250 →
  both = 80 →
  total = sci_fi + lit - both →
  lit = 230 := by
sorry

end literary_readers_count_l1104_110439


namespace order_relation_l1104_110428

theorem order_relation (a b c d : ℝ) 
  (h1 : d - a < c - b) 
  (h2 : c - b < 0) 
  (h3 : d - b = c - a) : 
  d < c ∧ c < b ∧ b < a := by
  sorry

end order_relation_l1104_110428


namespace bookshelf_capacity_l1104_110476

theorem bookshelf_capacity (num_bookshelves : ℕ) (layers_per_bookshelf : ℕ) (books_per_layer : ℕ) 
  (h1 : num_bookshelves = 8) 
  (h2 : layers_per_bookshelf = 5) 
  (h3 : books_per_layer = 85) : 
  num_bookshelves * layers_per_bookshelf * books_per_layer = 3400 := by
  sorry

end bookshelf_capacity_l1104_110476


namespace triangle_base_length_l1104_110443

/-- Theorem: The base of a triangle with specific side lengths -/
theorem triangle_base_length (left_side right_side base : ℝ) : 
  left_side = 12 →
  right_side = left_side + 2 →
  left_side + right_side + base = 50 →
  base = 24 := by
sorry

end triangle_base_length_l1104_110443


namespace log_relation_l1104_110417

theorem log_relation (p q : ℝ) : 
  p = Real.log 192 / Real.log 5 → 
  q = Real.log 12 / Real.log 3 → 
  p = (q * (Real.log 12 / Real.log 3 + 8/3)) / (Real.log 5 / Real.log 3) := by
sorry

end log_relation_l1104_110417


namespace irrational_between_neg_three_and_neg_two_l1104_110455

theorem irrational_between_neg_three_and_neg_two :
  ∃ x : ℝ, Irrational x ∧ -3 < x ∧ x < -2 := by sorry

end irrational_between_neg_three_and_neg_two_l1104_110455


namespace hotel_room_charge_comparison_l1104_110405

/-- Given the room charges for three hotels P, R, and G, prove that R's charge is 170% greater than G's. -/
theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.7 * R) 
  (h2 : P = G - 0.1 * G) : 
  (R - G) / G = 1.7 := by
  sorry

end hotel_room_charge_comparison_l1104_110405


namespace probability_of_sum_seven_l1104_110435

def standard_die := Finset.range 6
def special_die := Finset.range 7

def sum_of_dice (a : ℕ) (b : ℕ) : ℕ :=
  a + if b = 6 then 0 else b + 1

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (standard_die.product special_die).filter (λ p => sum_of_dice p.1 p.2 = 7)

theorem probability_of_sum_seven :
  (favorable_outcomes.card : ℚ) / ((standard_die.card * special_die.card) : ℚ) = 1 / 7 := by
  sorry

end probability_of_sum_seven_l1104_110435


namespace total_weight_of_cans_l1104_110462

theorem total_weight_of_cans (weights : List ℕ) (h : weights = [444, 459, 454, 459, 454, 454, 449, 454, 459, 464]) : 
  weights.sum = 4550 := by
  sorry

end total_weight_of_cans_l1104_110462


namespace abs_minus_one_eq_zero_l1104_110493

theorem abs_minus_one_eq_zero (a : ℝ) : |a| - 1 = 0 → a = 1 ∨ a = -1 := by
  sorry

end abs_minus_one_eq_zero_l1104_110493


namespace max_soap_boxes_in_carton_l1104_110422

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 6 }

/-- Theorem: The maximum number of soap boxes that can be placed in the carton is 250 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 250 := by
  sorry


end max_soap_boxes_in_carton_l1104_110422


namespace sarahs_friends_ages_sum_l1104_110424

theorem sarahs_friends_ages_sum :
  ∀ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 →  -- single-digit integers
    a ≠ b ∧ b ≠ c ∧ a ≠ c →     -- distinct
    a * b = 36 →                -- product of two ages is 36
    c ∣ 36 →                    -- third age is a factor of 36
    c ≠ a ∧ c ≠ b →             -- third age is not one of the first two
    a + b + c = 16 :=           -- sum of all three ages is 16
by sorry

end sarahs_friends_ages_sum_l1104_110424


namespace bert_sale_earnings_l1104_110452

/-- Calculates Bert's earnings from a sale given the selling price, markup, and tax rate. -/
def bertEarnings (sellingPrice markup taxRate : ℚ) : ℚ :=
  let purchasePrice := sellingPrice - markup
  let tax := taxRate * sellingPrice
  sellingPrice - tax - purchasePrice

/-- Theorem: Given a selling price of $90, a markup of $10, and a tax rate of 10%, Bert's earnings are $1. -/
theorem bert_sale_earnings :
  bertEarnings 90 10 (1/10) = 1 := by
  sorry

#eval bertEarnings 90 10 (1/10)

end bert_sale_earnings_l1104_110452


namespace quadratic_opens_downward_l1104_110474

def f (x : ℝ) := -x^2 + 3

theorem quadratic_opens_downward :
  ∃ (a : ℝ), ∀ (x : ℝ), x > a → f x < f a :=
sorry

end quadratic_opens_downward_l1104_110474


namespace program_output_l1104_110485

def program (a b : ℕ) : ℕ × ℕ :=
  let a' := a + b
  let b' := b * a'
  (a', b')

theorem program_output : program 1 3 = (4, 12) := by
  sorry

end program_output_l1104_110485


namespace polynomial_division_remainder_l1104_110468

/-- The polynomial to be divided -/
def P (z : ℝ) : ℝ := 4*z^4 - 3*z^3 + 2*z^2 - 16*z + 9

/-- The divisor polynomial -/
def D (z : ℝ) : ℝ := 4*z + 6

/-- The theorem stating that the remainder of P(z) divided by D(z) is 173/12 -/
theorem polynomial_division_remainder :
  ∃ Q : ℝ → ℝ, ∀ z : ℝ, P z = D z * Q z + 173/12 :=
sorry

end polynomial_division_remainder_l1104_110468


namespace problem_statement_l1104_110432

theorem problem_statement (a b : ℝ) : 
  |a - 3| + (b + 4)^2 = 0 → (a + b)^2003 = -1 := by
  sorry

end problem_statement_l1104_110432


namespace total_houses_l1104_110461

theorem total_houses (dogs : ℕ) (cats : ℕ) (both : ℕ) (h1 : dogs = 40) (h2 : cats = 30) (h3 : both = 10) :
  dogs + cats - both = 60 := by
  sorry

end total_houses_l1104_110461


namespace curve_transformation_l1104_110447

theorem curve_transformation (x y : ℝ) : 
  y = Real.sin (π / 2 + 2 * x) → 
  y = -Real.cos (5 * π / 6 - 3 * ((2 / 3) * x - π / 18)) := by
sorry

end curve_transformation_l1104_110447


namespace line_intersects_circle_l1104_110426

/-- The line y = k(x-2) + 4 intersects the curve y = √(4-x²) if and only if k ∈ [3/4, +∞) -/
theorem line_intersects_circle (k : ℝ) : 
  (∃ x y : ℝ, y = k * (x - 2) + 4 ∧ y = Real.sqrt (4 - x^2)) ↔ k ≥ 3/4 := by
  sorry

end line_intersects_circle_l1104_110426


namespace sum_of_abc_equals_45_l1104_110400

-- Define a triangle with side lengths 3, 7, and x
structure Triangle where
  x : ℝ
  side1 : ℝ := 3
  side2 : ℝ := 7
  side3 : ℝ := x

-- Define the property of angles in arithmetic progression
def anglesInArithmeticProgression (t : Triangle) : Prop := sorry

-- Define the sum of possible values of x
def sumOfPossibleX (t : Triangle) : ℝ := sorry

-- Define a, b, and c as positive integers
def a : ℕ+ := sorry
def b : ℕ+ := sorry
def c : ℕ+ := sorry

-- Theorem statement
theorem sum_of_abc_equals_45 (t : Triangle) 
  (h1 : anglesInArithmeticProgression t) 
  (h2 : sumOfPossibleX t = a + Real.sqrt b + Real.sqrt c) : 
  a + b + c = 45 := by sorry

end sum_of_abc_equals_45_l1104_110400


namespace alyssa_soccer_games_l1104_110444

theorem alyssa_soccer_games (this_year last_year next_year total : ℕ) 
  (h1 : this_year = 11)
  (h2 : last_year = 13)
  (h3 : next_year = 15)
  (h4 : total = 39)
  (h5 : this_year + last_year + next_year = total) : 
  this_year - (total - (last_year + next_year)) = 0 :=
by sorry

end alyssa_soccer_games_l1104_110444


namespace power_relation_l1104_110407

theorem power_relation (a : ℝ) (m n : ℤ) (h1 : a^m = 3) (h2 : a^n = 2) :
  a^(m - 2*n) = 3/4 := by
sorry

end power_relation_l1104_110407


namespace baseball_cards_count_l1104_110472

theorem baseball_cards_count (num_friends : ℕ) (cards_per_friend : ℕ) : 
  num_friends = 5 → cards_per_friend = 91 → num_friends * cards_per_friend = 455 := by
  sorry

end baseball_cards_count_l1104_110472


namespace points_on_line_l1104_110482

/-- Three points lie on the same line if and only if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- The theorem states that if the given points are collinear, then a = -7/23 -/
theorem points_on_line (a : ℝ) : 
  collinear (3, -5) (-a + 2, 3) (2*a + 3, 2) → a = -7/23 := by
  sorry

end points_on_line_l1104_110482


namespace max_value_abc_expression_l1104_110423

theorem max_value_abc_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ (1 : ℝ) / 4 := by
sorry

end max_value_abc_expression_l1104_110423


namespace notebook_cost_l1104_110471

theorem notebook_cost (total_cost cover_cost notebook_cost : ℚ) : 
  total_cost = 3.5 →
  notebook_cost = cover_cost + 2 →
  total_cost = notebook_cost + cover_cost →
  notebook_cost = 2.75 := by
sorry

end notebook_cost_l1104_110471


namespace percentage_same_grade_l1104_110404

def total_students : ℕ := 40
def students_with_all_As : ℕ := 3
def students_with_all_Bs : ℕ := 2
def students_with_all_Cs : ℕ := 6
def students_with_all_Ds : ℕ := 1

def students_with_same_grade : ℕ := 
  students_with_all_As + students_with_all_Bs + students_with_all_Cs + students_with_all_Ds

theorem percentage_same_grade : 
  (students_with_same_grade : ℚ) / total_students * 100 = 30 := by
  sorry

end percentage_same_grade_l1104_110404
