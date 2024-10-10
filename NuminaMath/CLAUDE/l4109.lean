import Mathlib

namespace dice_probability_l4109_410968

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Predicate to check if a number is a multiple of 5 -/
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k, n = 5 * k

/-- The set of all possible outcomes when rolling num_dice dice -/
def all_outcomes : Finset (Fin num_dice → Fin num_sides) := sorry

/-- The set of outcomes where at least one die shows an even number -/
def even_product_outcomes : Finset (Fin num_dice → Fin num_sides) := sorry

/-- The set of outcomes where the sum of dice is a multiple of 5 -/
def sum_multiple_of_5_outcomes : Finset (Fin num_dice → Fin num_sides) := sorry

/-- The number of favorable outcomes (sum is multiple of 5 given product is even) -/
def a : ℕ := sorry

/-- The probability of the sum being a multiple of 5 given the product is even -/
theorem dice_probability : 
  (Finset.card (even_product_outcomes ∩ sum_multiple_of_5_outcomes) : ℚ) / 
  (Finset.card even_product_outcomes : ℚ) = 
  (a : ℚ) / ((num_sides ^ num_dice - (num_sides / 2) ^ num_dice) : ℚ) :=
sorry

end dice_probability_l4109_410968


namespace train_length_l4109_410913

/-- Calculates the length of a train given its speed, time to pass a station, and the station's length. -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (station_length : ℝ) :
  train_speed = 36 * (1000 / 3600) →
  time_to_pass = 45 →
  station_length = 200 →
  train_speed * time_to_pass - station_length = 250 := by
  sorry

end train_length_l4109_410913


namespace negation_of_square_positive_equals_zero_l4109_410996

theorem negation_of_square_positive_equals_zero :
  (¬ ∀ m : ℝ, m > 0 → m^2 = 0) ↔ (∀ m : ℝ, m ≤ 0 → m^2 ≠ 0) :=
by sorry

end negation_of_square_positive_equals_zero_l4109_410996


namespace parallel_lines_a_equals_3_l4109_410937

/-- Two lines in the x-y plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- The condition for two lines to be distinct (not coincident) -/
def distinct (l1 l2 : Line) : Prop := l1.intercept ≠ l2.intercept

theorem parallel_lines_a_equals_3 (a : ℝ) :
  let l1 : Line := { slope := a^2, intercept := 3*a - a^2 }
  let l2 : Line := { slope := 4*a - 3, intercept := 2 }
  parallel l1 l2 ∧ distinct l1 l2 → a = 3 :=
by sorry

end parallel_lines_a_equals_3_l4109_410937


namespace gum_pack_size_l4109_410966

theorem gum_pack_size (initial_peach : ℕ) (initial_mint : ℕ) (y : ℚ) 
  (h1 : initial_peach = 40)
  (h2 : initial_mint = 50)
  (h3 : y > 0) :
  (initial_peach - 2 * y) / initial_mint = initial_peach / (initial_mint + 3 * y) → 
  y = 10 / 3 := by
sorry

end gum_pack_size_l4109_410966


namespace equilateral_not_obtuse_l4109_410910

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define properties of a triangle
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angleA > 90 ∨ t.angleB > 90 ∨ t.angleC > 90

-- Theorem: An equilateral triangle cannot be obtuse
theorem equilateral_not_obtuse (t : Triangle) :
  t.isEquilateral → ¬t.isObtuse := by
  sorry

end equilateral_not_obtuse_l4109_410910


namespace baker_cakes_theorem_l4109_410959

def calculate_remaining_cakes (initial_cakes : ℕ) (sold_cakes : ℕ) (additional_cakes : ℕ) : ℕ :=
  initial_cakes - sold_cakes + additional_cakes

theorem baker_cakes_theorem (initial_cakes sold_cakes additional_cakes : ℕ) 
  (h1 : initial_cakes ≥ sold_cakes) :
  calculate_remaining_cakes initial_cakes sold_cakes additional_cakes = 
  initial_cakes - sold_cakes + additional_cakes :=
by
  sorry

end baker_cakes_theorem_l4109_410959


namespace no_obtuse_angle_at_center_l4109_410989

/-- Represents a point on a circle -/
structure CirclePoint where
  arc : Fin 3
  position : ℝ
  h_position : 0 ≤ position ∧ position < 2 * Real.pi / 3

/-- Represents a configuration of 6 points on a circle -/
def CircleConfiguration := Fin 6 → CirclePoint

/-- Checks if three points form an obtuse angle at the center -/
def has_obtuse_angle_at_center (config : CircleConfiguration) (p1 p2 p3 : Fin 6) : Prop :=
  ∃ θ, θ > Real.pi / 2 ∧
    θ = min (2 * Real.pi / 3) (abs ((config p2).position - (config p1).position) +
      abs ((config p3).position - (config p2).position) +
      abs ((config p1).position - (config p3).position))

/-- The main theorem statement -/
theorem no_obtuse_angle_at_center (config : CircleConfiguration) :
  ∀ p1 p2 p3 : Fin 6, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
  ¬(has_obtuse_angle_at_center config p1 p2 p3) :=
sorry

end no_obtuse_angle_at_center_l4109_410989


namespace tangent_product_30_60_l4109_410988

theorem tangent_product_30_60 (A B : Real) (hA : A = 30 * π / 180) (hB : B = 60 * π / 180) :
  (1 + Real.tan A) * (1 + Real.tan B) = (3 + 4 * Real.sqrt 3) / 3 := by
  sorry

end tangent_product_30_60_l4109_410988


namespace distinct_terms_in_expansion_l4109_410920

/-- The number of distinct terms in the expansion of (a+b+c+d)(e+f+g+h+i),
    given that terms involving the product of a and e, and b and f are identical
    and combine into a single term. -/
theorem distinct_terms_in_expansion : ℕ := by
  sorry

end distinct_terms_in_expansion_l4109_410920


namespace min_value_sum_of_fractions_l4109_410943

theorem min_value_sum_of_fractions (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 ∧
  (x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) = 3 * Real.sqrt 3 / 2 ↔ 
   x = Real.sqrt 3 / 3 ∧ y = Real.sqrt 3 / 3 ∧ z = Real.sqrt 3 / 3) :=
by sorry

end min_value_sum_of_fractions_l4109_410943


namespace fraction_to_decimal_l4109_410942

theorem fraction_to_decimal (h : 160 = 2^5 * 5) : 7 / 160 = 0.175 := by
  sorry

end fraction_to_decimal_l4109_410942


namespace article_sale_price_l4109_410979

/-- Proves that the selling price incurring a loss equal to the profit from selling at 832 is 448,
    given the conditions stated in the problem. -/
theorem article_sale_price (cp : ℝ) : 
  (832 - cp = cp - 448) →  -- Profit from selling at 832 equals loss when sold at unknown amount
  (768 - cp = 0.2 * cp) →  -- Sale price for 20% profit is 768
  (448 : ℝ) = 832 - 2 * cp := by
  sorry

#check article_sale_price

end article_sale_price_l4109_410979


namespace projectile_trajectory_area_l4109_410964

open Real

/-- The area enclosed by the locus of highest points of projectile trajectories. -/
theorem projectile_trajectory_area (v g : ℝ) (h : ℝ := v^2 / (8 * g)) : 
  ∃ (area : ℝ), area = (3 * π / 32) * (v^4 / g^2) :=
by sorry

end projectile_trajectory_area_l4109_410964


namespace problem_solution_l4109_410975

theorem problem_solution (a b : ℝ) 
  (h1 : 2 * a - 1 = 9) 
  (h2 : 3 * a + b - 1 = 16) : 
  a + 2 * b = 9 := by
  sorry

end problem_solution_l4109_410975


namespace homework_time_distribution_l4109_410955

theorem homework_time_distribution (total_time : ℕ) (math_percent : ℚ) (science_percent : ℚ) 
  (h1 : total_time = 150)
  (h2 : math_percent = 30 / 100)
  (h3 : science_percent = 40 / 100) :
  total_time - (math_percent * total_time + science_percent * total_time) = 45 := by
  sorry

end homework_time_distribution_l4109_410955


namespace production_value_range_l4109_410944

-- Define the production value function
def f (x : ℝ) : ℝ := x * (220 - 2 * x)

-- Define the theorem
theorem production_value_range :
  ∀ x : ℝ, f x ≥ 6000 ↔ 50 < x ∧ x < 60 :=
by sorry

end production_value_range_l4109_410944


namespace first_car_speed_l4109_410982

/-- Represents the scenario of two cars traveling between points A and B -/
structure CarScenario where
  distance_AB : ℝ
  delay : ℝ
  speed_second_car : ℝ
  speed_first_car : ℝ

/-- Checks if the given scenario satisfies all conditions -/
def satisfies_conditions (s : CarScenario) : Prop :=
  s.distance_AB = 40 ∧
  s.delay = 1/3 ∧
  s.speed_second_car = 45 ∧
  ∃ (meeting_point : ℝ),
    0 < meeting_point ∧ meeting_point < s.distance_AB ∧
    (meeting_point / s.speed_second_car + s.delay = meeting_point / s.speed_first_car) ∧
    (s.distance_AB / s.speed_first_car = 
      meeting_point / s.speed_second_car + s.delay + meeting_point / s.speed_second_car + meeting_point / (2 * s.speed_second_car))

/-- The main theorem stating that if a scenario satisfies all conditions, 
    then the speed of the first car must be 30 km/h -/
theorem first_car_speed (s : CarScenario) :
  satisfies_conditions s → s.speed_first_car = 30 := by
  sorry

end first_car_speed_l4109_410982


namespace interest_rate_increase_l4109_410997

theorem interest_rate_increase (initial_rate : ℝ) (increase_percentage : ℝ) (final_rate : ℝ) : 
  initial_rate = 8.256880733944953 →
  increase_percentage = 10 →
  final_rate = initial_rate * (1 + increase_percentage / 100) →
  final_rate = 9.082568807339448 := by
sorry

end interest_rate_increase_l4109_410997


namespace logical_equivalences_l4109_410926

theorem logical_equivalences (A B C : Prop) : 
  ((A ∧ (B ∨ C) ↔ (A ∧ B) ∨ (A ∧ C)) ∧ 
   (A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C))) := by
  sorry

end logical_equivalences_l4109_410926


namespace decryption_theorem_l4109_410915

/-- Represents a character in the Russian alphabet --/
inductive RussianChar : Type
| A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z | AA | AB | AC | AD | AE | AF | AG

/-- Represents an encrypted message --/
def EncryptedMessage := List Char

/-- Represents a decrypted message --/
def DecryptedMessage := List RussianChar

/-- Converts a base-7 number to base-10 --/
def baseSevenToTen (n : Int) : Int :=
  sorry

/-- Applies Caesar cipher shift to a character --/
def applyCaesarShift (c : Char) (shift : Int) : RussianChar :=
  sorry

/-- Decrypts a message using Caesar cipher and base-7 to base-10 conversion --/
def decryptMessage (msg : EncryptedMessage) (shift : Int) : DecryptedMessage :=
  sorry

/-- Checks if a decrypted message is valid Russian text --/
def isValidRussianText (msg : DecryptedMessage) : Prop :=
  sorry

/-- The main theorem: decrypting the messages with shift 22 results in valid Russian text --/
theorem decryption_theorem (messages : List EncryptedMessage) :
  ∀ msg ∈ messages, isValidRussianText (decryptMessage msg 22) :=
  sorry

end decryption_theorem_l4109_410915


namespace total_defective_rate_l4109_410998

/-- Given two workers x and y who check products, with known defective rates and
    the fraction of products checked by worker y, prove the total defective rate. -/
theorem total_defective_rate 
  (defective_rate_x : ℝ) 
  (defective_rate_y : ℝ) 
  (fraction_checked_by_y : ℝ) 
  (h1 : defective_rate_x = 0.005) 
  (h2 : defective_rate_y = 0.008) 
  (h3 : fraction_checked_by_y = 0.5) 
  (h4 : fraction_checked_by_y ≥ 0 ∧ fraction_checked_by_y ≤ 1) : 
  defective_rate_x * (1 - fraction_checked_by_y) + defective_rate_y * fraction_checked_by_y = 0.0065 := by
  sorry

#check total_defective_rate

end total_defective_rate_l4109_410998


namespace extremum_and_inequality_l4109_410999

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - 1) * Real.log (x + a)

theorem extremum_and_inequality (h : ∀ a > 0, ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f a x ≤ f a 0) :
  (∃ a > 0, (deriv (f a)) 0 = 0) ∧
  (∀ x ≥ 0, f 1 x ≥ x^2) :=
sorry

end extremum_and_inequality_l4109_410999


namespace min_female_participants_l4109_410994

theorem min_female_participants (male_students female_students : ℕ) 
  (total_participants : ℕ) (h1 : male_students = 22) (h2 : female_students = 18) 
  (h3 : total_participants = (male_students + female_students) * 60 / 100) :
  ∃ (female_participants : ℕ), 
    female_participants ≥ 2 ∧ 
    female_participants ≤ female_students ∧
    female_participants + male_students ≥ total_participants :=
by
  sorry

end min_female_participants_l4109_410994


namespace quadrilateral_properties_exist_l4109_410980

noncomputable def quadrilateral_properties (a b c d t : ℝ) : Prop :=
  ∃ (α β γ δ ε : ℝ) (e f : ℝ),
    α + β + γ + δ = 2 * Real.pi ∧
    a * d * Real.sin α + b * c * Real.sin γ = 2 * t ∧
    a * b * Real.sin β + c * d * Real.sin δ = 2 * t ∧
    e^2 = a^2 + b^2 - 2*a*b * Real.cos β ∧
    e^2 = c^2 + d^2 - 2*c*d * Real.cos δ ∧
    f^2 = a^2 + d^2 - 2*a*d * Real.cos α ∧
    f^2 = b^2 + c^2 - 2*b*c * Real.cos γ ∧
    t = (1/2) * e * f * Real.sin ε

theorem quadrilateral_properties_exist (a b c d t : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (ht : t > 0) :
  quadrilateral_properties a b c d t :=
sorry

end quadrilateral_properties_exist_l4109_410980


namespace large_pizzas_purchased_l4109_410927

/-- Represents the number of slices in a small pizza -/
def small_pizza_slices : ℕ := 4

/-- Represents the number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- Represents the number of small pizzas purchased -/
def small_pizzas_purchased : ℕ := 3

/-- Represents the total number of slices consumed by all people -/
def total_slices_consumed : ℕ := 18

/-- Represents the number of slices left over -/
def slices_left_over : ℕ := 10

theorem large_pizzas_purchased :
  ∃ (n : ℕ), n * large_pizza_slices + small_pizzas_purchased * small_pizza_slices =
    total_slices_consumed + slices_left_over ∧ n = 2 := by
  sorry

end large_pizzas_purchased_l4109_410927


namespace integral_f_equals_three_l4109_410991

-- Define the function to be integrated
def f (x : ℝ) : ℝ := 2 - |1 - x|

-- State the theorem
theorem integral_f_equals_three :
  ∫ x in (0 : ℝ)..2, f x = 3 := by sorry

end integral_f_equals_three_l4109_410991


namespace similar_triangles_side_length_l4109_410934

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def SimilarTriangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_side_length 
  (P Q R S T U : ℝ × ℝ) 
  (h_similar : SimilarTriangles {P, Q, R} {S, T, U}) 
  (h_PQ : dist P Q = 10) 
  (h_QR : dist Q R = 15) 
  (h_ST : dist S T = 6) : 
  dist T U = 9 := by sorry

end similar_triangles_side_length_l4109_410934


namespace winner_received_62_percent_l4109_410946

/-- Represents an election with two candidates -/
structure Election where
  winner_votes : ℕ
  winning_margin : ℕ

/-- Calculates the percentage of votes received by the winner -/
def winner_percentage (e : Election) : ℚ :=
  (e.winner_votes : ℚ) / ((e.winner_votes + (e.winner_votes - e.winning_margin)) : ℚ) * 100

/-- Theorem stating that in the given election scenario, the winner received 62% of votes -/
theorem winner_received_62_percent :
  let e : Election := { winner_votes := 775, winning_margin := 300 }
  winner_percentage e = 62 := by sorry

end winner_received_62_percent_l4109_410946


namespace root_exists_in_interval_l4109_410935

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 6

-- State the theorem
theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ), f c = 0 :=
by
  have h1 : Continuous f := sorry
  have h2 : f (-2) < 0 := sorry
  have h3 : f (-1) > 0 := sorry
  sorry -- Proof omitted

end root_exists_in_interval_l4109_410935


namespace equation_system_solution_l4109_410983

def equation_system (x y z : ℝ) : Prop :=
  x^2 + y + z = 1 ∧ x + y^2 + z = 1 ∧ x + y + z^2 = 1

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 0, 0), (0, 1, 0), (0, 0, 1), 
   (-1 - Real.sqrt 2, -1 - Real.sqrt 2, -1 - Real.sqrt 2),
   (-1 + Real.sqrt 2, -1 + Real.sqrt 2, -1 + Real.sqrt 2)}

theorem equation_system_solution :
  ∀ x y z : ℝ, equation_system x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end equation_system_solution_l4109_410983


namespace gcd_1260_924_l4109_410957

theorem gcd_1260_924 : Nat.gcd 1260 924 = 84 := by
  sorry

end gcd_1260_924_l4109_410957


namespace hidden_primes_average_l4109_410904

/-- Given three cards with numbers on both sides, this theorem proves that
    the average of the hidden prime numbers is 46/3, given the conditions
    specified in the problem. -/
theorem hidden_primes_average (card1_visible card2_visible card3_visible : ℕ)
  (card1_hidden card2_hidden card3_hidden : ℕ)
  (h1 : card1_visible = 68)
  (h2 : card2_visible = 39)
  (h3 : card3_visible = 57)
  (h4 : Nat.Prime card1_hidden)
  (h5 : Nat.Prime card2_hidden)
  (h6 : Nat.Prime card3_hidden)
  (h7 : card1_visible + card1_hidden = card2_visible + card2_hidden)
  (h8 : card2_visible + card2_hidden = card3_visible + card3_hidden) :
  (card1_hidden + card2_hidden + card3_hidden : ℚ) / 3 = 46 / 3 := by
  sorry

#eval (46 : ℚ) / 3

end hidden_primes_average_l4109_410904


namespace sqrt_equation_solution_l4109_410956

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end sqrt_equation_solution_l4109_410956


namespace inverse_function_constraint_l4109_410950

theorem inverse_function_constraint (a b c d h : ℝ) : 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) →
  (∀ x, x ∈ Set.range (fun x => (a * (x + h) + b) / (c * (x + h) + d)) →
    (a * ((a * (x + h) + b) / (c * (x + h) + d) + h) + b) / 
    (c * ((a * (x + h) + b) / (c * (x + h) + d) + h) + d) = x) →
  a + d - 2 * c * h = 0 := by
sorry

end inverse_function_constraint_l4109_410950


namespace system_solution_unique_l4109_410986

theorem system_solution_unique :
  ∃! (x y : ℝ), 2 * x + 3 * y = 7 ∧ 4 * x - 3 * y = 5 :=
by
  -- Proof goes here
  sorry

end system_solution_unique_l4109_410986


namespace vector_parallelism_l4109_410903

theorem vector_parallelism (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, x]
  (∃ (k : ℝ), (a + b) = k • (a - b)) → x = -4 := by
sorry

end vector_parallelism_l4109_410903


namespace composition_equality_l4109_410960

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + a
def g (a : ℝ) (x : ℝ) : ℝ := a*x^2 + 1

-- State the theorem
theorem composition_equality (a : ℝ) :
  ∃ b : ℝ, ∀ x : ℝ, f a (g a x) = a^2*x^4 + 5*a*x^2 + b → b = 6 + a :=
by sorry

end composition_equality_l4109_410960


namespace equal_roots_quadratic_l4109_410938

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) → 
  m = 4 := by
  sorry

end equal_roots_quadratic_l4109_410938


namespace inverse_function_point_l4109_410954

open Real

theorem inverse_function_point (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  tan (π / 3) - f 2 = Real.sqrt 3 - 1 / 3 →
  Function.invFun f (1 / 3) - π / 2 = 2 - π / 2 := by
sorry

end inverse_function_point_l4109_410954


namespace ab_inequality_and_minimum_l4109_410912

theorem ab_inequality_and_minimum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 8) : 
  (a * b ≥ 16) ∧ 
  (a + 4 * b ≥ 17) ∧ 
  (a + 4 * b = 17 → a = 7) := by
sorry

end ab_inequality_and_minimum_l4109_410912


namespace triangle_properties_l4109_410928

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c * Real.cos t.A = 2 * t.b * Real.cos t.A ∧
  t.b + t.c = Real.sqrt 10 ∧
  t.a = 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 3 ∧ (1 / 2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 / 2 := by
  sorry

end triangle_properties_l4109_410928


namespace cubic_coefficient_b_is_zero_l4109_410916

-- Define the cubic function
def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem cubic_coefficient_b_is_zero
  (a b c d : ℝ) :
  (g a b c d (-2) = 0) →
  (g a b c d 0 = 0) →
  (g a b c d 2 = 0) →
  (g a b c d 1 = -1) →
  b = 0 := by
  sorry

end cubic_coefficient_b_is_zero_l4109_410916


namespace R_share_is_1295_l4109_410974

/-- Represents the capital invested by a partner -/
structure Capital where
  amount : ℚ
  is_positive : amount > 0

/-- Represents the investment scenario of the four partners -/
structure InvestmentScenario where
  P : Capital
  Q : Capital
  R : Capital
  S : Capital
  ratio_PQ : 4 * P.amount = 6 * Q.amount
  ratio_QR : 6 * Q.amount = 10 * R.amount
  S_investment : S.amount = P.amount + Q.amount
  total_profit : ℚ
  profit_is_positive : total_profit > 0

/-- Calculates the share of profit for partner R -/
def calculate_R_share (scenario : InvestmentScenario) : ℚ :=
  let total_capital := scenario.P.amount + scenario.Q.amount + scenario.R.amount + scenario.S.amount
  (scenario.total_profit * scenario.R.amount) / total_capital

/-- Theorem stating that R's share of profit is 1295 given the investment scenario -/
theorem R_share_is_1295 (scenario : InvestmentScenario) (h : scenario.total_profit = 12090) :
  calculate_R_share scenario = 1295 := by
  sorry

end R_share_is_1295_l4109_410974


namespace quadratic_function_bounds_l4109_410901

theorem quadratic_function_bounds (a : ℝ) (m : ℝ) : 
  a ≠ 0 → a < 0 → 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → 
    -2 ≤ a * x^2 + 2 * x + 1 ∧ a * x^2 + 2 * x + 1 ≤ 2) →
  m = 3 := by
sorry

end quadratic_function_bounds_l4109_410901


namespace linear_function_not_in_quadrant_I_l4109_410970

/-- A linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Definition of Quadrant I in the Cartesian plane -/
def QuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The main theorem stating that the given linear function does not pass through Quadrant I -/
theorem linear_function_not_in_quadrant_I (f : LinearFunction) 
  (h1 : f.m = -2)
  (h2 : f.b = -1) : 
  ¬ ∃ (x y : ℝ), y = f.m * x + f.b ∧ QuadrantI x y :=
sorry

end linear_function_not_in_quadrant_I_l4109_410970


namespace canteen_distance_l4109_410945

theorem canteen_distance (road_distance : ℝ) (perpendicular_distance : ℝ) 
  (hypotenuse_distance : ℝ) (canteen_distance : ℝ) :
  road_distance = 400 ∧ 
  perpendicular_distance = 300 ∧ 
  hypotenuse_distance = 500 ∧
  canteen_distance^2 = perpendicular_distance^2 + (road_distance - canteen_distance)^2 →
  canteen_distance = 312.5 := by
sorry

end canteen_distance_l4109_410945


namespace y_in_terms_of_x_l4109_410939

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
sorry

end y_in_terms_of_x_l4109_410939


namespace real_and_equal_roots_l4109_410971

/-- The quadratic equation in the problem -/
def quadratic_equation (k x : ℝ) : ℝ := 3 * x^2 - k * x + 2 * x + 15

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ := (k - 2)^2 - 4 * 3 * 15

theorem real_and_equal_roots (k : ℝ) :
  (∃ x : ℝ, quadratic_equation k x = 0 ∧
    ∀ y : ℝ, quadratic_equation k y = 0 → y = x) ↔
  (k = 6 * Real.sqrt 5 + 2 ∨ k = -6 * Real.sqrt 5 + 2) :=
sorry

end real_and_equal_roots_l4109_410971


namespace min_value_expression_l4109_410978

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a + b) / c + (a + c) / b + (b + c) / a ≥ 8 ∧
  (2 * (a + b) / c + (a + c) / b + (b + c) / a = 8 ↔ a = b ∧ b = c) :=
by sorry

end min_value_expression_l4109_410978


namespace parallel_vectors_imply_x_equals_5_l4109_410947

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Given vectors a and b, if they are parallel, then x = 5 -/
theorem parallel_vectors_imply_x_equals_5 :
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (2, 1)
  are_parallel a b → x = 5 := by
  sorry


end parallel_vectors_imply_x_equals_5_l4109_410947


namespace extrema_of_sum_l4109_410972

theorem extrema_of_sum (x y : ℝ) (h : x - 3 * Real.sqrt (x + 1) = 3 * Real.sqrt (y + 2) - y) :
  let P := x + y
  (9 + 3 * Real.sqrt 21) / 2 ≤ P ∧ P ≤ 9 + 3 * Real.sqrt 15 := by
  sorry

end extrema_of_sum_l4109_410972


namespace inequality_proof_l4109_410967

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end inequality_proof_l4109_410967


namespace quadratic_equation_solution_l4109_410940

theorem quadratic_equation_solution (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x : ℝ, x^2 + 14*x = 96 ∧ x > 0 ∧ x = Real.sqrt a - b) → a + b = 152 := by
  sorry

end quadratic_equation_solution_l4109_410940


namespace ticket_sales_revenue_l4109_410953

/-- The total money made from ticket sales given the conditions -/
def total_money_made (advance_price same_day_price total_tickets advance_tickets : ℕ) : ℕ :=
  advance_price * advance_tickets + same_day_price * (total_tickets - advance_tickets)

/-- Theorem stating that the total money made is $1600 under the given conditions -/
theorem ticket_sales_revenue : 
  total_money_made 20 30 60 20 = 1600 := by
  sorry

end ticket_sales_revenue_l4109_410953


namespace remainder_problem_l4109_410922

theorem remainder_problem (n : ℕ) (a b c d : ℕ) : 
  n > 0 → 
  n = 102 * a + b → 
  n = 103 * c + d → 
  0 ≤ b → b < 102 → 
  0 ≤ d → d < 103 → 
  a + d = 20 → 
  b = 20 := by
  sorry

end remainder_problem_l4109_410922


namespace spell_casting_contest_orders_l4109_410973

/-- The number of different possible orders for a given number of competitors -/
def possibleOrders (n : ℕ) : ℕ := Nat.factorial n

/-- The number of competitors in the contest -/
def numberOfCompetitors : ℕ := 4

theorem spell_casting_contest_orders :
  possibleOrders numberOfCompetitors = 24 := by
  sorry

end spell_casting_contest_orders_l4109_410973


namespace final_salary_proof_l4109_410941

def original_salary : ℝ := 20000
def reduction_rate : ℝ := 0.1
def increase_rate : ℝ := 0.1

def salary_after_changes (s : ℝ) (r : ℝ) (i : ℝ) : ℝ :=
  s * (1 - r) * (1 + i)

theorem final_salary_proof :
  salary_after_changes original_salary reduction_rate increase_rate = 19800 := by
  sorry

end final_salary_proof_l4109_410941


namespace x_squared_y_squared_range_l4109_410932

theorem x_squared_y_squared_range (x y : ℝ) (h : x^2 + y^2 = 2*x) :
  0 ≤ x^2 * y^2 ∧ x^2 * y^2 ≤ 27/16 := by
  sorry

end x_squared_y_squared_range_l4109_410932


namespace line_parallel_from_perpendicular_to_parallel_planes_l4109_410963

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (non_coincident : Line → Line → Prop)
variable (plane_non_coincident : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_from_perpendicular_to_parallel_planes
  (m n : Line) (α β : Plane)
  (h_non_coincident : non_coincident m n)
  (h_plane_non_coincident : plane_non_coincident α β)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_β : perpendicular n β)
  (h_α_parallel_β : plane_parallel α β) :
  parallel m n :=
sorry

end line_parallel_from_perpendicular_to_parallel_planes_l4109_410963


namespace video_upvotes_l4109_410923

theorem video_upvotes (up_to_down_ratio : Rat) (down_votes : ℕ) (up_votes : ℕ) : 
  up_to_down_ratio = 9 / 2 → down_votes = 4 → up_votes = 18 := by
  sorry

end video_upvotes_l4109_410923


namespace minute_hand_angle_for_2h40m_l4109_410948

/-- Represents the angle turned by the minute hand of a clock -/
def minuteHandAngle (hours : ℝ) (minutes : ℝ) : ℝ :=
  -(hours * 360 + minutes * 6)

/-- 
Theorem: When the hour hand of a clock moves for 2 hours and 40 minutes 
in a clockwise direction, the minute hand turns through an angle of -960°
-/
theorem minute_hand_angle_for_2h40m : 
  minuteHandAngle 2 40 = -960 := by
  sorry

end minute_hand_angle_for_2h40m_l4109_410948


namespace inequality_count_l4109_410985

theorem inequality_count (x y a b : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_x_lt_a : x < a) (h_y_lt_b : y < b) : 
  (((x + y < a + b) ∧ (x * y < a * b) ∧ (x / y < a / b)) ∧ 
   ¬(∀ x y a b, x > 0 → y > 0 → a > 0 → b > 0 → x < a → y < b → x - y < a - b)) := by
  sorry

end inequality_count_l4109_410985


namespace lucy_lovely_age_difference_l4109_410914

/-- Represents the ages of Lucy and Lovely at different points in time -/
structure Ages where
  lucy_current : ℕ
  lovely_current : ℕ
  years_ago : ℕ

/-- Conditions of the problem -/
def problem_conditions (a : Ages) : Prop :=
  a.lucy_current = 50 ∧
  a.lucy_current - a.years_ago = 3 * (a.lovely_current - a.years_ago) ∧
  a.lucy_current + 10 = 2 * (a.lovely_current + 10)

/-- Theorem stating the solution to the problem -/
theorem lucy_lovely_age_difference :
  ∃ (a : Ages), problem_conditions a ∧ a.years_ago = 5 :=
sorry

end lucy_lovely_age_difference_l4109_410914


namespace smallest_angle_measure_l4109_410984

-- Define a triangle with angles in 2:3:4 ratio
def triangle_with_ratio (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  b = (3/2) * a ∧ c = 2 * a ∧
  a + b + c = 180

-- Theorem statement
theorem smallest_angle_measure (a b c : ℝ) 
  (h : triangle_with_ratio a b c) : a = 40 := by
  sorry

end smallest_angle_measure_l4109_410984


namespace correction_is_11y_l4109_410977

/-- The correction needed when y quarters are mistakenly counted as nickels
    and y pennies are mistakenly counted as dimes -/
def correction (y : ℕ) : ℤ :=
  let quarter_value : ℕ := 25
  let nickel_value : ℕ := 5
  let penny_value : ℕ := 1
  let dime_value : ℕ := 10
  let quarter_nickel_diff : ℕ := quarter_value - nickel_value
  let dime_penny_diff : ℕ := dime_value - penny_value
  (quarter_nickel_diff * y : ℤ) - (dime_penny_diff * y : ℤ)

theorem correction_is_11y (y : ℕ) : correction y = 11 * y :=
  sorry

end correction_is_11y_l4109_410977


namespace anna_spending_l4109_410992

/-- Anna's spending problem -/
theorem anna_spending (original : ℚ) (left : ℚ) (h1 : original = 32) (h2 : left = 24) :
  (original - left) / original = 1 / 4 := by
  sorry

end anna_spending_l4109_410992


namespace car_journey_time_l4109_410907

theorem car_journey_time (distance : ℝ) (new_speed : ℝ) (initial_time : ℝ) : 
  distance = 360 →
  new_speed = 40 →
  distance / new_speed = (3/2) * initial_time →
  initial_time = 6 := by
sorry

end car_journey_time_l4109_410907


namespace P_root_nature_l4109_410981

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^5 - 4*x^4 - 6*x^3 - x + 8

-- Theorem stating that P(x) has no negative roots and at least one positive root
theorem P_root_nature :
  (∀ x < 0, P x ≠ 0) ∧ (∃ x > 0, P x = 0) := by
  sorry


end P_root_nature_l4109_410981


namespace playlist_song_length_l4109_410962

theorem playlist_song_length 
  (n_short_songs : ℕ) 
  (short_song_length : ℕ) 
  (n_long_songs : ℕ) 
  (total_duration : ℕ) 
  (additional_time_needed : ℕ) 
  (h1 : n_short_songs = 10)
  (h2 : short_song_length = 3)
  (h3 : n_long_songs = 15)
  (h4 : total_duration = 100)
  (h5 : additional_time_needed = 40) :
  ∃ (long_song_length : ℚ),
    long_song_length = 14/3 ∧ 
    n_short_songs * short_song_length + n_long_songs * long_song_length = total_duration := by
  sorry

end playlist_song_length_l4109_410962


namespace existence_of_midpoint_with_odd_double_coordinates_l4109_410929

/-- A point in the xy-plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A sequence of 1993 distinct points with the required properties -/
def PointSequence : Type :=
  { ps : Fin 1993 → IntPoint //
    (∀ i j, i ≠ j → ps i ≠ ps j) ∧  -- points are distinct
    (∀ i : Fin 1992, ∀ p : IntPoint,
      p ≠ ps i ∧ p ≠ ps (i + 1) →
      ¬∃ (t : ℚ), 0 < t ∧ t < 1 ∧
        p.x = (1 - t) * (ps i).x + t * (ps (i + 1)).x ∧
        p.y = (1 - t) * (ps i).y + t * (ps (i + 1)).y) }

theorem existence_of_midpoint_with_odd_double_coordinates (ps : PointSequence) :
    ∃ i : Fin 1992, ∃ qx qy : ℚ,
      (2 * qx).num % 2 = 1 ∧
      (2 * qy).num % 2 = 1 ∧
      qx = ((ps.val i).x + (ps.val (i + 1)).x) / 2 ∧
      qy = ((ps.val i).y + (ps.val (i + 1)).y) / 2 := by
  sorry

end existence_of_midpoint_with_odd_double_coordinates_l4109_410929


namespace problem_solution_l4109_410918

open Real

noncomputable def f (x : ℝ) : ℝ :=
  ((1 + cos (2*x))^2 - 2*cos (2*x) - 1) / (sin (π/4 + x) * sin (π/4 - x))

noncomputable def g (x : ℝ) : ℝ :=
  (1/2) * f x + sin (2*x)

theorem problem_solution :
  (f (-11*π/12) = Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc 0 (π/4), g x ≤ Real.sqrt 2) ∧
  (∀ x ∈ Set.Icc 0 (π/4), g x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 (π/4), g x = Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (π/4), g x = 1) :=
by sorry

end problem_solution_l4109_410918


namespace freds_dimes_l4109_410925

/-- Fred's dime problem -/
theorem freds_dimes (initial_dimes borrowed_dimes : ℕ) 
  (h1 : initial_dimes = 7)
  (h2 : borrowed_dimes = 3) :
  initial_dimes - borrowed_dimes = 4 := by
  sorry

end freds_dimes_l4109_410925


namespace ratio_problem_l4109_410924

/-- Given two positive integers A and B, where A < B, if A = 36 and LCM(A, B) = 180, then A:B = 1:5 -/
theorem ratio_problem (A B : ℕ) (h1 : 0 < A) (h2 : A < B) (h3 : A = 36) (h4 : Nat.lcm A B = 180) :
  A * 5 = B * 1 := by
  sorry

end ratio_problem_l4109_410924


namespace harry_work_hours_l4109_410993

/-- Given the payment conditions for Harry and James, prove that if James worked 41 hours
    and they were paid the same amount, then Harry worked 39 hours. -/
theorem harry_work_hours (x : ℝ) (h : ℝ) :
  let harry_pay := 24 * x + (h - 24) * 1.5 * x
  let james_pay := 24 * x + (41 - 24) * 2 * x
  harry_pay = james_pay →
  h = 39 := by
  sorry

end harry_work_hours_l4109_410993


namespace line_vector_proof_l4109_410936

def line_vector (t : ℚ) : ℚ × ℚ × ℚ := sorry

theorem line_vector_proof :
  (line_vector (-2) = (2, 6, 16)) ∧
  (line_vector 1 = (0, -1, -2)) ∧
  (line_vector 4 = (-2, -8, -18)) →
  (line_vector 0 = (2/3, 4/3, 4)) ∧
  (line_vector 5 = (-8, -19, -26)) := by sorry

end line_vector_proof_l4109_410936


namespace intersection_implies_a_value_l4109_410965

def set_A (a : ℝ) : Set ℝ := {-2, 3*a-1, a^2-3}
def set_B (a : ℝ) : Set ℝ := {a-2, a-1, a+1}

theorem intersection_implies_a_value (a : ℝ) :
  set_A a ∩ set_B a = {-2} → a = -3 :=
by
  sorry

end intersection_implies_a_value_l4109_410965


namespace farm_animals_l4109_410949

theorem farm_animals (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 5 * initial_cows →
  (initial_horses - 15) / (initial_cows + 15) = 17 / 7 →
  (initial_horses - 15) - (initial_cows + 15) = 50 := by
sorry

end farm_animals_l4109_410949


namespace methane_moles_required_l4109_410917

-- Define the chemical species involved
structure ChemicalSpecies where
  methane : ℝ
  chlorine : ℝ
  chloromethane : ℝ
  hydrochloric_acid : ℝ

-- Define the reaction conditions
def reaction_conditions (reactants products : ChemicalSpecies) : Prop :=
  reactants.chlorine = 2 ∧
  products.chloromethane = 2 ∧
  products.hydrochloric_acid = 2

-- Define the stoichiometric relationship
def stoichiometric_relationship (reactants products : ChemicalSpecies) : Prop :=
  reactants.methane = products.chloromethane ∧
  reactants.methane = products.hydrochloric_acid

-- Theorem statement
theorem methane_moles_required 
  (reactants products : ChemicalSpecies) 
  (h_conditions : reaction_conditions reactants products) 
  (h_stoichiometry : stoichiometric_relationship reactants products) : 
  reactants.methane = 2 := by
  sorry

end methane_moles_required_l4109_410917


namespace sin_five_pi_sixths_minus_two_alpha_l4109_410951

theorem sin_five_pi_sixths_minus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -1 / 3 := by
  sorry

end sin_five_pi_sixths_minus_two_alpha_l4109_410951


namespace lcm_of_48_and_180_l4109_410930

theorem lcm_of_48_and_180 : Nat.lcm 48 180 = 720 := by
  sorry

end lcm_of_48_and_180_l4109_410930


namespace coffee_blend_type_A_quantity_l4109_410990

/-- Represents the cost and quantity of coffee types in Amanda's Coffee Shop blend --/
structure CoffeeBlend where
  typeA_cost : ℝ
  typeB_cost : ℝ
  typeA_quantity : ℝ
  typeB_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the quantity of type A coffee in the blend --/
theorem coffee_blend_type_A_quantity (blend : CoffeeBlend) 
  (h1 : blend.typeA_cost = 4.60)
  (h2 : blend.typeB_cost = 5.95)
  (h3 : blend.typeB_quantity = 2 * blend.typeA_quantity)
  (h4 : blend.total_cost = 511.50)
  (h5 : blend.total_cost = blend.typeA_cost * blend.typeA_quantity + blend.typeB_cost * blend.typeB_quantity) :
  blend.typeA_quantity = 31 := by
  sorry


end coffee_blend_type_A_quantity_l4109_410990


namespace quadratic_vertex_form_l4109_410961

/-- Given a quadratic expression 3x^2 + 9x - 24, when written in the form a(x - h)^2 + k, h = -1.5 -/
theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a k : ℝ), 3*x^2 + 9*x - 24 = a*(x - (-1.5))^2 + k :=
by sorry

end quadratic_vertex_form_l4109_410961


namespace total_age_calculation_l4109_410906

def family_gathering (K : ℕ) : Prop :=
  let father_age : ℕ := 60
  let mother_age : ℕ := father_age - 2
  let brother_age : ℕ := father_age / 2
  let sister_age : ℕ := 40
  let elder_cousin_age : ℕ := brother_age + 2 * sister_age
  let younger_cousin_age : ℕ := elder_cousin_age / 2 + 3
  let grandmother_age : ℕ := 3 * mother_age - 5
  let T : ℕ := father_age + mother_age + brother_age + sister_age + 
               elder_cousin_age + younger_cousin_age + grandmother_age + K
  T = 525 + K

theorem total_age_calculation (K : ℕ) : family_gathering K :=
  sorry

end total_age_calculation_l4109_410906


namespace coin_arrangement_concyclic_l4109_410976

-- Define the circles (coins)
variable (O₁ O₂ O₃ O₄ : ℝ × ℝ)  -- Centers of the circles
variable (r₁ r₂ r₃ r₄ : ℝ)      -- Radii of the circles

-- Define the points of intersection
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define the property of being concyclic
def concyclic (p q r s : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem coin_arrangement_concyclic :
  concyclic A B C D :=
sorry

end coin_arrangement_concyclic_l4109_410976


namespace total_routes_bristol_to_carlisle_l4109_410969

theorem total_routes_bristol_to_carlisle :
  let bristol_to_birmingham : ℕ := 8
  let birmingham_to_manchester : ℕ := 5
  let manchester_to_sheffield : ℕ := 4
  let sheffield_to_newcastle : ℕ := 3
  let newcastle_to_carlisle : ℕ := 2
  bristol_to_birmingham * birmingham_to_manchester * manchester_to_sheffield * sheffield_to_newcastle * newcastle_to_carlisle = 960 := by
  sorry

end total_routes_bristol_to_carlisle_l4109_410969


namespace fourth_guard_theorem_l4109_410909

/-- Represents a rectangular facility with guards at each corner -/
structure Facility :=
  (length : ℝ)
  (width : ℝ)
  (guard_distance : ℝ)

/-- Calculates the distance run by the fourth guard -/
def fourth_guard_distance (f : Facility) : ℝ :=
  2 * (f.length + f.width) - f.guard_distance

/-- Theorem stating the distance run by the fourth guard -/
theorem fourth_guard_theorem (f : Facility) 
  (h1 : f.length = 200)
  (h2 : f.width = 300)
  (h3 : f.guard_distance = 850) :
  fourth_guard_distance f = 150 := by
  sorry

#eval fourth_guard_distance { length := 200, width := 300, guard_distance := 850 }

end fourth_guard_theorem_l4109_410909


namespace root_equation_problem_l4109_410921

theorem root_equation_problem (m r s a b : ℝ) : 
  (a^2 - m*a + 4 = 0) →
  (b^2 - m*b + 4 = 0) →
  ((a^2 + 1/b)^2 - r*(a^2 + 1/b) + s = 0) →
  ((b^2 + 1/a)^2 - r*(b^2 + 1/a) + s = 0) →
  s = m + 16.25 := by
sorry

end root_equation_problem_l4109_410921


namespace medical_team_selection_count_l4109_410987

theorem medical_team_selection_count : ∀ (m f k l : ℕ), 
  m = 6 → f = 5 → k = 2 → l = 1 →
  (m.choose k) * (f.choose l) = 75 :=
by sorry

end medical_team_selection_count_l4109_410987


namespace divisibility_by_1897_l4109_410905

theorem divisibility_by_1897 (n : ℕ) : 
  (1897 : ℤ) ∣ (2903^n - 803^n - 464^n + 261^n) := by
sorry

end divisibility_by_1897_l4109_410905


namespace complex_modulus_equation_l4109_410931

theorem complex_modulus_equation :
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (9 + t * Complex.I) = 15 ∧ t = 12 := by
  sorry

end complex_modulus_equation_l4109_410931


namespace investment_income_l4109_410995

theorem investment_income
  (total_investment : ℝ)
  (first_investment : ℝ)
  (first_rate : ℝ)
  (second_rate : ℝ)
  (h1 : total_investment = 8000)
  (h2 : first_investment = 3000)
  (h3 : first_rate = 0.085)
  (h4 : second_rate = 0.064) :
  first_investment * first_rate + (total_investment - first_investment) * second_rate = 575 := by
  sorry

end investment_income_l4109_410995


namespace weight_difference_l4109_410958

/-- Given the weights of Heather and Emily, prove the difference in their weights -/
theorem weight_difference (heather_weight emily_weight : ℕ) 
  (h1 : heather_weight = 87)
  (h2 : emily_weight = 9) :
  heather_weight - emily_weight = 78 := by
  sorry

end weight_difference_l4109_410958


namespace complex_equation_sum_l4109_410908

theorem complex_equation_sum (a b : ℝ) :
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a + b = 1 := by
  sorry

end complex_equation_sum_l4109_410908


namespace min_square_size_and_unused_area_l4109_410952

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- The shapes contained within the larger square -/
def contained_shapes : List Rectangle := [
  { width := 2, height := 2 },  -- 2x2 square
  { width := 1, height := 3 },  -- 1x3 rectangle
  { width := 2, height := 1 }   -- 2x1 rectangle
]

/-- Theorem: The minimum side length of the containing square is 5,
    and the minimum unused area is 16 -/
theorem min_square_size_and_unused_area :
  let min_side := 5
  let total_area := min_side * min_side
  let shapes_area := (contained_shapes.map Rectangle.area).sum
  let unused_area := total_area - shapes_area
  (∀ side : ℕ, side ≥ min_side → 
    side * side - shapes_area ≥ unused_area) ∧
  unused_area = 16 := by
  sorry

end min_square_size_and_unused_area_l4109_410952


namespace polynomial_identities_l4109_410902

theorem polynomial_identities (x y : ℝ) : 
  ((x + y)^3 - x^3 - y^3 = 3*x*y*(x + y)) ∧ 
  ((x + y)^5 - x^5 - y^5 = 5*x*y*(x + y)*(x^2 + x*y + y^2)) ∧ 
  ((x + y)^7 - x^7 - y^7 = 7*x*y*(x + y)*(x^2 + x*y + y^2)^2) := by
  sorry

end polynomial_identities_l4109_410902


namespace sin_330_degrees_l4109_410911

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l4109_410911


namespace triangle_third_side_length_l4109_410900

theorem triangle_third_side_length 
  (a b : ℝ) 
  (angle : ℝ) 
  (ha : a = 9) 
  (hb : b = 10) 
  (hangle : angle = Real.pi * 3 / 4) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2 * a * b * Real.cos angle ∧ 
            c = Real.sqrt (181 + 90 * Real.sqrt 2) := by
  sorry

end triangle_third_side_length_l4109_410900


namespace bicyclist_effective_speed_l4109_410919

/-- Calculates the effective speed of a bicyclist considering headwind -/
def effective_speed (initial_speed_ms : ℝ) (headwind_kmh : ℝ) : ℝ :=
  initial_speed_ms * 3.6 - headwind_kmh

/-- Proves that the effective speed of a bicyclist with an initial speed of 18 m/s
    and a headwind of 10 km/h is 54.8 km/h -/
theorem bicyclist_effective_speed :
  effective_speed 18 10 = 54.8 := by sorry

end bicyclist_effective_speed_l4109_410919


namespace solution_comparison_l4109_410933

theorem solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0)
  (h_sol : -q / p > -q' / p') : q / p < q' / p' := by
  sorry

end solution_comparison_l4109_410933
