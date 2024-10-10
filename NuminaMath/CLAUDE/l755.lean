import Mathlib

namespace cyclist_average_speed_l755_75570

/-- Proves that a cyclist traveling 136.4 km in 6 hours and 30 minutes has an average speed of approximately 5.83 m/s -/
theorem cyclist_average_speed :
  let distance_km : ℝ := 136.4
  let time_hours : ℝ := 6.5
  let distance_m : ℝ := distance_km * 1000
  let time_s : ℝ := time_hours * 3600
  let average_speed : ℝ := distance_m / time_s
  ∃ ε > 0, |average_speed - 5.83| < ε :=
by
  sorry

end cyclist_average_speed_l755_75570


namespace train_platform_ratio_l755_75531

/-- Proves that the ratio of train length to platform length is 1:1 given the specified conditions --/
theorem train_platform_ratio (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 216 * (1000 / 3600) →
  train_length = 1800 →
  crossing_time = 60 →
  ∃ (platform_length : ℝ), train_length / platform_length = 1 := by
  sorry


end train_platform_ratio_l755_75531


namespace ap_sum_100_l755_75542

/-- Given an arithmetic progression where:
    - The sum of the first 15 terms is 45
    - The sum of the first 85 terms is 255
    This theorem proves that the sum of the first 100 terms is 300. -/
theorem ap_sum_100 (a d : ℝ) 
  (sum_15 : (15 : ℝ) / 2 * (2 * a + (15 - 1) * d) = 45)
  (sum_85 : (85 : ℝ) / 2 * (2 * a + (85 - 1) * d) = 255) :
  (100 : ℝ) / 2 * (2 * a + (100 - 1) * d) = 300 :=
by sorry

end ap_sum_100_l755_75542


namespace rain_probability_theorem_l755_75579

/-- The probability of rain on each day -/
def rain_prob : ℚ := 2/3

/-- The number of consecutive days -/
def num_days : ℕ := 5

/-- The probability of no rain on a single day -/
def no_rain_prob : ℚ := 1 - rain_prob

/-- The probability of two consecutive dry days -/
def two_dry_days_prob : ℚ := no_rain_prob ^ 2

/-- The number of pairs of consecutive days in the given period -/
def num_pairs : ℕ := num_days - 1

theorem rain_probability_theorem :
  (no_rain_prob ^ num_days = 1/243) ∧
  (two_dry_days_prob * num_pairs = 4/9) := by
  sorry


end rain_probability_theorem_l755_75579


namespace circle_area_from_circumference_l755_75532

theorem circle_area_from_circumference (r : ℝ) (k : ℝ) : 
  (2 * π * r = 36 * π) → (π * r^2 = k * π) → k = 324 := by
  sorry

end circle_area_from_circumference_l755_75532


namespace parabola_equation_l755_75562

/-- A vertical line passing through a point (x₀, y₀) -/
structure VerticalLine where
  x₀ : ℝ
  y₀ : ℝ

/-- A parabola with vertical axis and vertex at origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := λ x y => y^2 = -2 * p * x

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (l : VerticalLine) (para : Parabola) :
  l.x₀ = 3/2 ∧ l.y₀ = 2 ∧ para.eq = λ x y => y^2 = -6*x := by sorry

end parabola_equation_l755_75562


namespace donation_to_first_home_l755_75509

theorem donation_to_first_home 
  (total_donation : ℝ) 
  (second_home_donation : ℝ) 
  (third_home_donation : ℝ) 
  (h1 : total_donation = 700)
  (h2 : second_home_donation = 225)
  (h3 : third_home_donation = 230) :
  total_donation - second_home_donation - third_home_donation = 245 :=
by sorry

end donation_to_first_home_l755_75509


namespace log_relation_l755_75537

theorem log_relation (a b : ℝ) (h1 : a = Real.log 256 / Real.log 4) (h2 : b = Real.log 27 / Real.log 3) :
  a = (4/3) * b := by sorry

end log_relation_l755_75537


namespace num_clips_property_l755_75533

/-- The number of clips on a curtain rod after k halving steps -/
def num_clips (k : ℕ) : ℕ :=
  2^(k-1) + 1

/-- The property that each interval has a middle clip -/
def has_middle_clip (n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → ∃ j : ℕ, j < n ∧ j > i ∧ j - i = (n - i) / 2

/-- The theorem stating that num_clips satisfies the middle clip property for all steps -/
theorem num_clips_property (k : ℕ) : 
  k > 0 → has_middle_clip (num_clips k) :=
sorry

end num_clips_property_l755_75533


namespace sqrt_sum_equality_l755_75585

theorem sqrt_sum_equality : 
  let a : ℕ := 49
  let b : ℕ := 64
  let c : ℕ := 100
  Real.sqrt a + Real.sqrt b + Real.sqrt c = 
    Real.sqrt (219 + Real.sqrt 10080 + Real.sqrt 12600 + Real.sqrt 35280) := by
  sorry

end sqrt_sum_equality_l755_75585


namespace maximum_value_of_x_plus_reciprocal_x_l755_75593

theorem maximum_value_of_x_plus_reciprocal_x (x : ℝ) :
  x < 0 → ∃ (max : ℝ), (∀ y, y < 0 → y + 1/y ≤ max) ∧ max = -2 :=
sorry


end maximum_value_of_x_plus_reciprocal_x_l755_75593


namespace sum_of_four_numbers_l755_75552

theorem sum_of_four_numbers (a b c d : ℤ) 
  (sum_abc : a + b + c = 415)
  (sum_abd : a + b + d = 442)
  (sum_acd : a + c + d = 396)
  (sum_bcd : b + c + d = 325) :
  a + b + c + d = 526 := by
sorry

end sum_of_four_numbers_l755_75552


namespace largest_five_digit_congruent_to_15_mod_24_l755_75527

theorem largest_five_digit_congruent_to_15_mod_24 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n ≡ 15 [MOD 24] → n ≤ 99999 :=
by sorry

end largest_five_digit_congruent_to_15_mod_24_l755_75527


namespace complement_of_P_l755_75529

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x | x - 1 < 0}

-- State the theorem
theorem complement_of_P : Set.compl P = {x : ℝ | x ≥ 1} := by sorry

end complement_of_P_l755_75529


namespace cubic_roots_sum_cubes_l755_75535

theorem cubic_roots_sum_cubes (α β γ : ℂ) : 
  (8 * α^3 + 2012 * α + 2013 = 0) →
  (8 * β^3 + 2012 * β + 2013 = 0) →
  (8 * γ^3 + 2012 * γ + 2013 = 0) →
  (α + β)^3 + (β + γ)^3 + (γ + α)^3 = 6039 / 8 := by
  sorry

end cubic_roots_sum_cubes_l755_75535


namespace correct_page_difference_l755_75506

/-- Calculates the difference in pages read between yesterday and today -/
def pagesDifference (totalPages yesterday tomorrow : ℕ) : ℕ :=
  yesterday - (totalPages - yesterday - tomorrow)

theorem correct_page_difference :
  pagesDifference 100 35 35 = 5 := by
  sorry

end correct_page_difference_l755_75506


namespace vector_addition_l755_75508

def a : Fin 2 → ℝ := ![3, 1]
def b : Fin 2 → ℝ := ![-2, 5]

theorem vector_addition : 2 • a + b = ![4, 7] := by sorry

end vector_addition_l755_75508


namespace max_value_parabola_l755_75538

theorem max_value_parabola :
  ∀ x : ℝ, 0 < x → x < 6 → x * (6 - x) ≤ 9 ∧ ∃ y : ℝ, 0 < y ∧ y < 6 ∧ y * (6 - y) = 9 := by
  sorry

end max_value_parabola_l755_75538


namespace quadratic_rational_root_implies_even_coefficient_l755_75540

theorem quadratic_rational_root_implies_even_coefficient
  (a b c : ℤ)
  (h_a_nonzero : a ≠ 0)
  (h_rational_root : ∃ (x : ℚ), a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end quadratic_rational_root_implies_even_coefficient_l755_75540


namespace spring_experiment_l755_75592

/-- Spring experiment data points -/
def spring_data : List (ℝ × ℝ) := [(0, 20), (1, 22), (2, 24), (3, 26), (4, 28), (5, 30)]

/-- The relationship between spring length y (in cm) and weight x (in kg) -/
def spring_relation (x y : ℝ) : Prop := y = 2 * x + 20

/-- Theorem stating that the spring_relation holds for all data points in spring_data -/
theorem spring_experiment :
  ∀ (point : ℝ × ℝ), point ∈ spring_data → spring_relation point.1 point.2 := by
  sorry

end spring_experiment_l755_75592


namespace product_of_good_sequences_is_good_l755_75518

/-- A sequence is a function from natural numbers to real numbers. -/
def Sequence := ℕ → ℝ

/-- The first derivative of a sequence. -/
def firstDerivative (a : Sequence) : Sequence :=
  λ n => a (n + 1) - a n

/-- The k-th derivative of a sequence. -/
def kthDerivative : ℕ → Sequence → Sequence
  | 0, a => a
  | k + 1, a => firstDerivative (kthDerivative k a)

/-- A sequence is good if it and all its derivatives consist of positive numbers. -/
def isGoodSequence (a : Sequence) : Prop :=
  ∀ k n, kthDerivative k a n > 0

/-- The element-wise product of two sequences. -/
def productSequence (a b : Sequence) : Sequence :=
  λ n => a n * b n

/-- Theorem: The element-wise product of two good sequences is also a good sequence. -/
theorem product_of_good_sequences_is_good (a b : Sequence) 
  (ha : isGoodSequence a) (hb : isGoodSequence b) : 
  isGoodSequence (productSequence a b) := by
  sorry

end product_of_good_sequences_is_good_l755_75518


namespace house_elves_do_not_exist_l755_75553

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (HouseElf : U → Prop)
variable (LovesPranks : U → Prop)
variable (LovesCleanlinessAndOrder : U → Prop)

-- State the theorem
theorem house_elves_do_not_exist :
  (∀ x, HouseElf x → LovesPranks x) →
  (∀ x, HouseElf x → LovesCleanlinessAndOrder x) →
  (∀ x, LovesCleanlinessAndOrder x → ¬LovesPranks x) →
  ¬(∃ x, HouseElf x) :=
by
  sorry

end house_elves_do_not_exist_l755_75553


namespace rectangle_tiling_existence_l755_75578

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tiling of rectangles -/
def Tiling := List Rectangle

/-- Checks if a list of rectangles can tile a larger rectangle -/
def canTile (tiles : Tiling) (target : Rectangle) : Prop := sorry

/-- The specific tiles we're allowed to use -/
def allowedTiles : Tiling := [Rectangle.mk 4 6, Rectangle.mk 5 7]

/-- The theorem stating the existence of N and that 840 is a valid value -/
theorem rectangle_tiling_existence :
  ∃ (N : ℕ), ∀ (m n : ℕ), m > N → n > N →
    canTile allowedTiles (Rectangle.mk m n) ∧ canTile allowedTiles (Rectangle.mk 841 841) := by
  sorry

#check rectangle_tiling_existence

end rectangle_tiling_existence_l755_75578


namespace clothes_pricing_l755_75580

/-- Given a total spend and a price relation between shirt and trousers,
    prove the individual costs of the shirt and trousers. -/
theorem clothes_pricing (total : ℕ) (shirt_price trousers_price : ℕ) 
    (h1 : total = 185)
    (h2 : shirt_price = 2 * trousers_price + 5)
    (h3 : total = shirt_price + trousers_price) :
    trousers_price = 60 ∧ shirt_price = 125 := by
  sorry

end clothes_pricing_l755_75580


namespace museum_art_count_l755_75582

theorem museum_art_count (total : ℕ) (asian : ℕ) (egyptian : ℕ) (european : ℕ) 
  (h1 : total = 2500)
  (h2 : asian = 465)
  (h3 : egyptian = 527)
  (h4 : european = 320) :
  total - (asian + egyptian + european) = 1188 := by
  sorry

end museum_art_count_l755_75582


namespace common_terms_k_polygonal_fermat_l755_75517

/-- k-polygonal number sequence -/
def kPolygonalSeq (k : ℕ) (n : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

/-- Fermat number sequence -/
def fermatSeq (m : ℕ) : ℕ :=
  2^(2^m) + 1

/-- Proposition: The only positive integers k > 2 for which there exist common terms
    between the k-polygonal numbers sequence and the Fermat numbers sequence are 3 and 5 -/
theorem common_terms_k_polygonal_fermat :
  {k : ℕ | k > 2 ∧ ∃ (n m : ℕ), kPolygonalSeq k n = fermatSeq m} = {3, 5} := by
  sorry

end common_terms_k_polygonal_fermat_l755_75517


namespace water_jar_problem_l755_75560

theorem water_jar_problem (small_jar large_jar : ℝ) (h1 : small_jar > 0) (h2 : large_jar > 0) 
  (h3 : small_jar ≠ large_jar) (water : ℝ) (h4 : water > 0)
  (h5 : water / small_jar = 1 / 7) (h6 : water / large_jar = 1 / 6) :
  (2 * water) / large_jar = 1 / 3 := by
sorry

end water_jar_problem_l755_75560


namespace ratio_problem_l755_75565

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) :
  x / y = 11 / 6 := by
sorry

end ratio_problem_l755_75565


namespace positive_integers_equality_l755_75534

theorem positive_integers_equality (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (4 * a * b - 1) ∣ ((4 * a^2 - 1)^2) → a = b := by
  sorry

end positive_integers_equality_l755_75534


namespace curve_symmetric_about_origin_l755_75516

/-- A curve defined by the equation xy - x^2 = 1 -/
def curve (x y : ℝ) : Prop := x * y - x^2 = 1

/-- Symmetry about the origin for the curve -/
theorem curve_symmetric_about_origin :
  ∀ x y : ℝ, curve x y ↔ curve (-x) (-y) :=
sorry

end curve_symmetric_about_origin_l755_75516


namespace lollipops_per_boy_l755_75576

theorem lollipops_per_boy (total_candies : ℕ) (total_children : ℕ) 
  (h1 : total_candies = 90)
  (h2 : total_children = 40)
  (h3 : ∃ (num_lollipops : ℕ), num_lollipops = total_candies / 3)
  (h4 : ∃ (num_candy_canes : ℕ), num_candy_canes = total_candies - total_candies / 3)
  (h5 : ∃ (num_girls : ℕ), num_girls = (total_candies - total_candies / 3) / 2)
  (h6 : ∃ (num_boys : ℕ), num_boys = total_children - (total_candies - total_candies / 3) / 2) :
  (total_candies / 3) / (total_children - (total_candies - total_candies / 3) / 2) = 3 := by
  sorry

end lollipops_per_boy_l755_75576


namespace handshakes_in_specific_gathering_l755_75504

/-- Represents a gathering of people -/
structure Gathering where
  total : Nat
  group1 : Nat
  group2 : Nat
  group1_knows_each_other : Bool
  group2_knows_no_one : Bool

/-- Calculates the number of handshakes in a gathering -/
def count_handshakes (g : Gathering) : Nat :=
  if g.group1_knows_each_other && g.group2_knows_no_one then
    (g.group2 * (g.total - 1)) / 2
  else
    0  -- This case is not relevant for our specific problem

theorem handshakes_in_specific_gathering :
  let g : Gathering := {
    total := 30,
    group1 := 20,
    group2 := 10,
    group1_knows_each_other := true,
    group2_knows_no_one := true
  }
  count_handshakes g = 145 := by
  sorry

end handshakes_in_specific_gathering_l755_75504


namespace unique_sine_solution_l755_75548

theorem unique_sine_solution : ∃! x : Real, 0 ≤ x ∧ x < Real.pi ∧ Real.sin x = -0.45 := by
  sorry

end unique_sine_solution_l755_75548


namespace work_completion_l755_75598

/-- Represents the total amount of work in man-days -/
def total_work : ℕ := 10 * 80

/-- The number of days it takes for the second group to complete the work -/
def days_second_group : ℕ := 40

/-- Calculates the number of men needed to complete the work in a given number of days -/
def men_needed (days : ℕ) : ℕ := total_work / days

theorem work_completion :
  men_needed days_second_group = 20 :=
sorry

end work_completion_l755_75598


namespace semiperimeter_radius_sum_eq_legs_sum_l755_75586

/-- A right triangle with legs a and b, hypotenuse c, semiperimeter p, and inscribed circle radius r -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  r : ℝ
  right_angle : c^2 = a^2 + b^2
  semiperimeter : p = (a + b + c) / 2
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The sum of the semiperimeter and the radius of the inscribed circle is equal to the sum of the legs -/
theorem semiperimeter_radius_sum_eq_legs_sum (t : RightTriangle) : t.p + t.r = t.a + t.b := by
  sorry

end semiperimeter_radius_sum_eq_legs_sum_l755_75586


namespace subtract_point_five_from_47_point_two_l755_75561

theorem subtract_point_five_from_47_point_two : 47.2 - 0.5 = 46.7 := by
  sorry

end subtract_point_five_from_47_point_two_l755_75561


namespace mathematics_letter_probability_l755_75515

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of distinct letters in 'MATHEMATICS' -/
def distinct_letters : ℕ := 8

/-- The word we're considering -/
def word : String := "MATHEMATICS"

/-- Theorem: The probability of randomly selecting a letter from the alphabet
    that appears in 'MATHEMATICS' is 4/13 -/
theorem mathematics_letter_probability :
  (distinct_letters : ℚ) / (alphabet_size : ℚ) = 4 / 13 := by
  sorry

end mathematics_letter_probability_l755_75515


namespace number_relationship_l755_75581

theorem number_relationship (n : ℚ) : n = 25 / 3 → (6 * n - 10) - 3 * n = 15 := by
  sorry

end number_relationship_l755_75581


namespace relationship_abc_l755_75557

theorem relationship_abc (a b c : ℝ) :
  (∃ u v : ℝ, u - v = a ∧ u^2 - v^2 = b ∧ u^3 - v^3 = c) →
  3 * b^2 + a^4 = 4 * a * c := by
sorry

end relationship_abc_l755_75557


namespace ball_weight_problem_l755_75591

theorem ball_weight_problem (R W : ℚ) 
  (eq1 : 7 * R + 5 * W = 43)
  (eq2 : 5 * R + 7 * W = 47) :
  4 * R + 8 * W = 49 := by
  sorry

end ball_weight_problem_l755_75591


namespace perpendicular_parallel_perpendicular_perpendicular_parallel_planes_l755_75599

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_perpendicular
  (α : Plane) (m n : Line)
  (h1 : parallel_line_plane m α)
  (h2 : perpendicular_line_plane n α) :
  perpendicular_line_line m n :=
sorry

-- Theorem 2
theorem perpendicular_parallel_planes
  (α β : Plane) (m : Line)
  (h1 : perpendicular_line_plane m α)
  (h2 : parallel_plane_plane α β) :
  perpendicular_line_plane m β :=
sorry

end perpendicular_parallel_perpendicular_perpendicular_parallel_planes_l755_75599


namespace cyclic_quadrilateral_theorem_l755_75528

-- Define the points
variable (A B C D P Q E : Point)

-- Define the conditions
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry
def is_inside (P Q : Point) (A B C D : Point) : Prop := sorry
def is_cyclic_quadrilateral (P Q D A : Point) : Prop := sorry
def point_on_line (E P Q : Point) : Prop := sorry
def angle_eq (P A E Q D : Point) : Prop := sorry

-- State the theorem
theorem cyclic_quadrilateral_theorem 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_inside P Q A B C D)
  (h3 : is_cyclic_quadrilateral P Q D A)
  (h4 : is_cyclic_quadrilateral Q P B C)
  (h5 : point_on_line E P Q)
  (h6 : angle_eq P A E Q D)
  (h7 : angle_eq P B E Q C) :
  is_cyclic_quadrilateral A B C D :=
sorry

end cyclic_quadrilateral_theorem_l755_75528


namespace robbie_win_probability_l755_75502

/-- A special six-sided die where rolling number x is x times as likely as rolling a 1 -/
structure SpecialDie :=
  (prob_one : ℝ)
  (sum_to_one : prob_one * (1 + 2 + 3 + 4 + 5 + 6) = 1)

/-- The game where two players roll the special die three times each -/
def Game (d : SpecialDie) :=
  { score : ℕ × ℕ // score.1 ≤ 18 ∧ score.2 ≤ 18 }

/-- The probability of rolling a specific number on the special die -/
def prob_roll (d : SpecialDie) (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 6 then n * d.prob_one else 0

/-- The probability of Robbie winning given the current game state -/
def prob_robbie_win (d : SpecialDie) (g : Game d) : ℝ :=
  sorry

theorem robbie_win_probability (d : SpecialDie) (g : Game d) 
  (h1 : g.val.1 = 8) (h2 : g.val.2 = 10) : 
  prob_robbie_win d g = 55 / 441 :=
sorry

end robbie_win_probability_l755_75502


namespace candied_grape_price_l755_75510

-- Define the number of candied apples
def num_apples : ℕ := 15

-- Define the price of each candied apple
def price_apple : ℚ := 2

-- Define the number of candied grapes
def num_grapes : ℕ := 12

-- Define the total revenue
def total_revenue : ℚ := 48

-- Define the price of each candied grape
def price_grape : ℚ := 1.5

theorem candied_grape_price :
  price_grape * num_grapes + price_apple * num_apples = total_revenue :=
by sorry

end candied_grape_price_l755_75510


namespace valid_allocations_count_l755_75584

/-- The number of student volunteers --/
def num_students : ℕ := 5

/-- The number of display boards --/
def num_boards : ℕ := 2

/-- A function that calculates the number of ways to allocate students to display boards --/
def allocation_schemes (n : ℕ) (k : ℕ) (min_per_board : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid allocation schemes --/
theorem valid_allocations_count :
  allocation_schemes num_students num_boards 2 = 12 :=
sorry

end valid_allocations_count_l755_75584


namespace coefficient_a2_l755_75545

/-- Given z = 1/2 + (√3/2)i and (x-z)^4 = a₀x^4 + a₁x^3 + a₂x^2 + a₃x + a₄, prove that a₂ = -3 + 3√3i. -/
theorem coefficient_a2 (z : ℂ) (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  z = (1 : ℂ) / 2 + (Complex.I * Real.sqrt 3) / 2 →
  (∀ x : ℂ, (x - z)^4 = a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = -3 + 3 * Complex.I * Real.sqrt 3 := by
  sorry

end coefficient_a2_l755_75545


namespace complementary_angles_difference_l755_75568

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- angles are complementary
  a = 4 * b → -- ratio of measures is 4:1
  abs (a - b) = 54 := by sorry

end complementary_angles_difference_l755_75568


namespace no_solution_greater_than_two_l755_75549

theorem no_solution_greater_than_two (n : ℕ) (h : n > 2) :
  ¬ (3^(n-1) + 5^(n-1) ∣ 3^n + 5^n) := by
  sorry

end no_solution_greater_than_two_l755_75549


namespace integer_power_sum_l755_75563

theorem integer_power_sum (x : ℝ) (h : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
sorry

end integer_power_sum_l755_75563


namespace sum_of_squares_theorem_l755_75569

/-- Given a geometric sequence {a_n} where the sum of its first n terms is 2^n - 1,
    this function computes the sum of the first n terms of the sequence {a_n^2}. -/
def sum_of_squares (n : ℕ) : ℚ :=
  (4^n - 1) / 3

/-- The sum of the first n terms of the original geometric sequence {a_n}. -/
def sum_of_original (n : ℕ) : ℕ :=
  2^n - 1

theorem sum_of_squares_theorem (n : ℕ) :
  sum_of_squares n = (4^n - 1) / 3 :=
by sorry

end sum_of_squares_theorem_l755_75569


namespace triangle_side_length_l755_75594

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 4)
  (h2 : t.b = 5)
  (h3 : t.S = 5 * Real.sqrt 3) :
  t.c = Real.sqrt 21 ∨ t.c = Real.sqrt 61 := by
  sorry


end triangle_side_length_l755_75594


namespace family_boys_count_l755_75521

/-- A family where one child has 3 brothers and 6 sisters, and another child has 4 brothers and 5 sisters -/
structure Family where
  total_children : ℕ
  child1_brothers : ℕ
  child1_sisters : ℕ
  child2_brothers : ℕ
  child2_sisters : ℕ
  h1 : child1_brothers = 3
  h2 : child1_sisters = 6
  h3 : child2_brothers = 4
  h4 : child2_sisters = 5

/-- The number of boys in the family -/
def num_boys (f : Family) : ℕ := f.child1_brothers + 1

theorem family_boys_count (f : Family) : num_boys f = 4 := by
  sorry

end family_boys_count_l755_75521


namespace vanya_finished_first_l755_75547

/-- Represents a participant in the competition -/
structure Participant where
  name : String
  predicted_position : Nat
  actual_position : Nat

/-- The competition setup and results -/
structure Competition where
  participants : List Participant
  vanya : Participant

/-- Axioms for the competition -/
axiom all_positions_different (c : Competition) :
  ∀ p1 p2 : Participant, p1 ∈ c.participants → p2 ∈ c.participants → p1 ≠ p2 →
    p1.actual_position ≠ p2.actual_position

axiom vanya_predicted_last (c : Competition) :
  c.vanya.predicted_position = c.participants.length

axiom others_worse_than_predicted (c : Competition) :
  ∀ p : Participant, p ∈ c.participants → p ≠ c.vanya →
    p.actual_position > p.predicted_position

/-- Theorem: Vanya must have finished first -/
theorem vanya_finished_first (c : Competition) :
  c.vanya.actual_position = 1 :=
sorry

end vanya_finished_first_l755_75547


namespace one_absent_two_present_probability_l755_75525

def absent_probability : ℚ := 1 / 20

def present_probability : ℚ := 1 - absent_probability

def probability_one_absent_two_present (p q : ℚ) : ℚ := 3 * p * q * q

theorem one_absent_two_present_probability : 
  probability_one_absent_two_present absent_probability present_probability = 1083 / 8000 := by
  sorry

end one_absent_two_present_probability_l755_75525


namespace parallel_vectors_k_value_l755_75571

/-- Given vectors a and b, if (-2a + b) is parallel to (a + kb), then k = -1/2 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (-3, 1)) 
  (h2 : b = (1, -2)) 
  (h_parallel : ∃ (t : ℝ), t • (-2 • a + b) = (a + k • b)) :
  k = -1/2 := by
sorry

end parallel_vectors_k_value_l755_75571


namespace meet_once_l755_75588

/-- Represents the meeting scenario between Michael and the garbage truck --/
structure MeetingScenario where
  michaelSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ
  initialDistance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def numberOfMeetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once (scenario : MeetingScenario) 
  (h1 : scenario.michaelSpeed = 4)
  (h2 : scenario.truckSpeed = 8)
  (h3 : scenario.pailDistance = 300)
  (h4 : scenario.truckStopTime = 45)
  (h5 : scenario.initialDistance = 300) : 
  numberOfMeetings scenario = 1 :=
sorry

end meet_once_l755_75588


namespace set_intersection_problem_l755_75554

def A : Set ℝ := {1, 2, 6}
def B : Set ℝ := {2, 4}
def C : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}

theorem set_intersection_problem : (A ∪ B) ∩ C = {1, 2, 4} := by sorry

end set_intersection_problem_l755_75554


namespace surfer_wave_height_l755_75513

/-- Represents the height of the highest wave caught by a surfer. -/
def highest_wave (H : ℝ) : ℝ := 4 * H + 2

/-- Represents the height of the shortest wave caught by a surfer. -/
def shortest_wave (H : ℝ) : ℝ := H + 4

theorem surfer_wave_height (H : ℝ) 
  (h1 : shortest_wave H = 7 + 3) 
  (h2 : shortest_wave H = H + 4) : 
  highest_wave H = 26 := by
  sorry

end surfer_wave_height_l755_75513


namespace orange_pricing_theorem_l755_75595

/-- Represents a pricing scheme for oranges -/
structure PricingScheme where
  oranges : ℕ
  price : ℕ

/-- Calculates the minimum cost for buying a given number of oranges -/
def minCost (schemes : List PricingScheme) (totalOranges : ℕ) : ℕ :=
  sorry

/-- Calculates the average cost per orange -/
def avgCost (totalCost : ℕ) (totalOranges : ℕ) : ℚ :=
  sorry

theorem orange_pricing_theorem (schemes : List PricingScheme) (totalOranges : ℕ) :
  schemes = [⟨4, 12⟩, ⟨7, 30⟩] →
  totalOranges = 20 →
  avgCost (minCost schemes totalOranges) totalOranges = 3 := by
  sorry

end orange_pricing_theorem_l755_75595


namespace largest_base_for_12_cubed_digit_sum_base_8_sum_not_9_l755_75530

def base_representation (n : ℕ) (b : ℕ) : List ℕ :=
  sorry

def sum_of_digits (n : ℕ) (b : ℕ) : ℕ :=
  (base_representation n b).sum

def power_in_base (base : ℕ) (n : ℕ) (power : ℕ) : ℕ :=
  sorry

theorem largest_base_for_12_cubed_digit_sum :
  ∀ b : ℕ, b > 8 → sum_of_digits (power_in_base b 12 3) b = 9 :=
by sorry

theorem base_8_sum_not_9 :
  sum_of_digits (power_in_base 8 12 3) 8 ≠ 9 :=
by sorry

end largest_base_for_12_cubed_digit_sum_base_8_sum_not_9_l755_75530


namespace infinite_x₀_finite_values_l755_75507

/-- The function f(x) = 3x - x^2 -/
def f (x : ℝ) : ℝ := 3 * x - x^2

/-- The sequence x_n defined by x_n = f(x_{n-1}) -/
def seq (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f (seq x₀ n)

/-- A set is finite if it is empty or there exists a bijection with a finite segment of ℕ -/
def IsFiniteSet (S : Set ℝ) : Prop :=
  S = ∅ ∨ ∃ n : ℕ, ∃ h : Fin n → S, Function.Bijective h

/-- The set of values in the sequence starting from x₀ -/
def seqValues (x₀ : ℝ) : Set ℝ :=
  { x | ∃ n : ℕ, seq x₀ n = x }

/-- The theorem stating that infinitely many x₀ in [0, 3] lead to finite value sets -/
theorem infinite_x₀_finite_values :
  ∃ S : Set ℝ, S ⊆ Set.Icc 0 3 ∧ Set.Infinite S ∧ ∀ x₀ ∈ S, IsFiniteSet (seqValues x₀) := by
  sorry


end infinite_x₀_finite_values_l755_75507


namespace opposite_of_negative_2023_l755_75500

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end opposite_of_negative_2023_l755_75500


namespace pool_dimensions_l755_75574

/-- Represents the dimensions and costs of a rectangular open-top swimming pool. -/
structure Pool where
  shortSide : ℝ  -- Length of the shorter side of the rectangular bottom
  depth : ℝ      -- Depth of the pool
  bottomCost : ℝ -- Cost per square meter for constructing the bottom
  wallCost : ℝ   -- Cost per square meter for constructing the walls
  totalCost : ℝ  -- Total construction cost

/-- Calculates the total cost of constructing the pool. -/
def calculateCost (p : Pool) : ℝ :=
  p.bottomCost * p.shortSide * (2 * p.shortSide) + 
  p.wallCost * (p.shortSide + 2 * p.shortSide) * 2 * p.depth

/-- Theorem stating that the pool with given specifications has sides of 3m and 6m. -/
theorem pool_dimensions (p : Pool) 
  (h1 : p.depth = 2)
  (h2 : p.bottomCost = 200)
  (h3 : p.wallCost = 100)
  (h4 : p.totalCost = 7200)
  (h5 : calculateCost p = p.totalCost) :
  p.shortSide = 3 ∧ 2 * p.shortSide = 6 := by
  sorry

#check pool_dimensions

end pool_dimensions_l755_75574


namespace gcd_of_4557_1953_5115_l755_75575

theorem gcd_of_4557_1953_5115 : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end gcd_of_4557_1953_5115_l755_75575


namespace unknown_number_proof_l755_75514

theorem unknown_number_proof (x : ℝ) : x + 5 * 12 / (180 / 3) = 66 → x = 65 := by
  sorry

end unknown_number_proof_l755_75514


namespace triangle_side_product_l755_75587

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if (a+b)^2 - c^2 = 4 and C = 60°, then ab = 4/3 -/
theorem triangle_side_product (a b c : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : Real.cos (π/3) = 1/2) :
  a * b = 4/3 := by
  sorry

end triangle_side_product_l755_75587


namespace first_number_value_l755_75550

theorem first_number_value (x : ℝ) : x + 2 * (8 - 3) = 24.16 → x = 14.16 := by
  sorry

end first_number_value_l755_75550


namespace managers_salary_l755_75519

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) :
  num_employees = 20 →
  avg_salary = 1300 →
  salary_increase = 100 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) / (num_employees + 1) - avg_salary = salary_increase →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) - (num_employees * avg_salary) = 3400 :=
by
  sorry

end managers_salary_l755_75519


namespace shaded_area_calculation_l755_75589

theorem shaded_area_calculation (R : ℝ) (d : ℝ) (h1 : R = 10) (h2 : d = 8) : 
  let r : ℝ := Real.sqrt (R^2 - d^2)
  let large_circle_area : ℝ := π * R^2
  let small_circle_area : ℝ := 2 * π * r^2
  large_circle_area - small_circle_area = 28 * π :=
by sorry

end shaded_area_calculation_l755_75589


namespace hyperbolic_amplitude_properties_l755_75524

/-- Hyperbolic cosine -/
noncomputable def ch (x : ℝ) : ℝ := sorry

/-- Hyperbolic sine -/
noncomputable def sh (x : ℝ) : ℝ := sorry

/-- Hyperbolic tangent -/
noncomputable def th (x : ℝ) : ℝ := sorry

/-- Tangent -/
noncomputable def tg (α : ℝ) : ℝ := sorry

theorem hyperbolic_amplitude_properties (x α : ℝ) 
  (h1 : ch x ^ 2 - sh x ^ 2 = 1)
  (h2 : tg α = sh x / ch x) : 
  (ch x = 1 / Real.cos α) ∧ (th (x / 2) = tg (α / 2)) := by
  sorry

end hyperbolic_amplitude_properties_l755_75524


namespace product_301_52_base7_units_digit_l755_75511

theorem product_301_52_base7_units_digit (a b : ℕ) (ha : a = 301) (hb : b = 52) :
  (a * b) % 7 = 0 := by
  sorry

end product_301_52_base7_units_digit_l755_75511


namespace similar_triangle_perimeter_l755_75558

/-- Given an isosceles triangle and a similar triangle, calculates the perimeter of the larger triangle -/
theorem similar_triangle_perimeter (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a = b →
  c > a →
  c > b →
  d > c →
  (a + b + c) * (d / c) = 100 :=
by
  sorry

#check similar_triangle_perimeter

end similar_triangle_perimeter_l755_75558


namespace consecutive_integers_square_sum_l755_75503

theorem consecutive_integers_square_sum : 
  ∀ a : ℤ, a > 0 → 
  ((a - 1) * a * (a + 1) = 12 * (3 * a)) → 
  ((a - 1)^2 + a^2 + (a + 1)^2 = 77) := by
  sorry

end consecutive_integers_square_sum_l755_75503


namespace expression_simplification_l755_75559

theorem expression_simplification :
  (1 / ((1 / (Real.sqrt 2 + 1)) + (1 / (Real.sqrt 5 - 2)))) = 
  ((Real.sqrt 2 + Real.sqrt 5 - 1) / (6 + 2 * Real.sqrt 10)) := by
sorry

end expression_simplification_l755_75559


namespace sequence_difference_l755_75596

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

def geometric_sequence (g₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := g₁ * r^(n - 1 : ℕ)

theorem sequence_difference : 
  let a₁ := 3
  let a₂ := 11
  let g₁ := 2
  let g₂ := 10
  let d := a₂ - a₁
  let r := g₂ / g₁
  |arithmetic_sequence a₁ d 100 - geometric_sequence g₁ r 4| = 545 := by
sorry

end sequence_difference_l755_75596


namespace ron_eats_24_slices_l755_75583

/-- The number of pickle slices Sammy can eat -/
def sammy_slices : ℕ := 15

/-- Tammy can eat twice as many pickle slices as Sammy -/
def tammy_slices : ℕ := 2 * sammy_slices

/-- Ron eats 20% fewer pickle slices than Tammy -/
def ron_slices : ℕ := tammy_slices - (tammy_slices * 20 / 100)

/-- Theorem stating that Ron eats 24 pickle slices -/
theorem ron_eats_24_slices : ron_slices = 24 := by sorry

end ron_eats_24_slices_l755_75583


namespace garden_length_l755_75597

theorem garden_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = 2 + 3 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 100 →
  length = 38 := by
sorry

end garden_length_l755_75597


namespace hyperbola_properties_l755_75556

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 - x^2/4 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = x/2 ∨ y = -x/2

-- Theorem statement
theorem hyperbola_properties :
  ∀ (x y : ℝ),
  hyperbola_equation x y →
  (∃ (a : ℝ), hyperbola_equation 0 a) ∧
  (∀ (x' y' : ℝ), x' ≠ 0 → asymptote_equation x' y' → 
    ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > ε → 
      ∃ (x'' y'' : ℝ), hyperbola_equation x'' y'' ∧ 
      abs (x'' - x') < δ ∧ abs (y'' - y') < δ) :=
sorry

end hyperbola_properties_l755_75556


namespace jills_earnings_ratio_l755_75566

/-- Jill's earnings over three months --/
def total_earnings : ℝ := 1200

/-- Jill's daily earnings in the first month --/
def first_month_daily : ℝ := 10

/-- Number of days in each month --/
def days_per_month : ℕ := 30

/-- Ratio of second month's daily earnings to first month's daily earnings --/
def earnings_ratio : ℝ := 2

theorem jills_earnings_ratio : 
  ∃ (second_month_daily : ℝ),
    first_month_daily * days_per_month +
    second_month_daily * days_per_month +
    second_month_daily * (days_per_month / 2) = total_earnings ∧
    second_month_daily / first_month_daily = earnings_ratio :=
by sorry

end jills_earnings_ratio_l755_75566


namespace janes_trip_distance_l755_75543

theorem janes_trip_distance :
  ∀ (total_distance : ℝ),
  (1/4 : ℝ) * total_distance +     -- First part (highway)
  30 +                             -- Second part (city streets)
  (1/6 : ℝ) * total_distance       -- Third part (country roads)
  = total_distance                 -- Sum of all parts equals total distance
  →
  total_distance = 360/7 := by
sorry

end janes_trip_distance_l755_75543


namespace prime_arithmetic_sequence_ones_digit_l755_75522

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_arithmetic_sequence_ones_digit 
  (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hr : is_prime r) 
  (h_seq : q = p + 4 ∧ r = q + 4) 
  (h_p_gt_5 : p > 5) :
  ones_digit p = 3 ∨ ones_digit p = 9 :=
sorry

end prime_arithmetic_sequence_ones_digit_l755_75522


namespace no_rational_roots_odd_coeff_l755_75501

theorem no_rational_roots_odd_coeff (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Int.gcd p q = 1 ∧ a * p^2 + b * p * q + c * q^2 = 0 :=
sorry

end no_rational_roots_odd_coeff_l755_75501


namespace f_sum_2006_2007_l755_75590

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_periodic : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom f_1 : f 1 = 2

-- State the theorem
theorem f_sum_2006_2007 : f 2006 + f 2007 = 2 := by sorry

end f_sum_2006_2007_l755_75590


namespace max_investment_at_lower_rate_l755_75512

theorem max_investment_at_lower_rate 
  (total_investment : ℝ) 
  (low_rate high_rate : ℝ) 
  (min_interest : ℝ) 
  (h1 : total_investment = 25000)
  (h2 : low_rate = 0.07)
  (h3 : high_rate = 0.12)
  (h4 : min_interest = 2450) :
  let max_low_investment := 11000
  ∀ x : ℝ, 
    0 ≤ x ∧ 
    x ≤ total_investment ∧ 
    low_rate * x + high_rate * (total_investment - x) ≥ min_interest →
    x ≤ max_low_investment := by
sorry

end max_investment_at_lower_rate_l755_75512


namespace marble_return_condition_l755_75526

/-- Represents the motion of a marble on a horizontal table with elastic collision -/
structure MarbleMotion where
  v₀ : ℝ  -- Initial speed
  h : ℝ   -- Initial height
  D : ℝ   -- Distance to vertical wall
  g : ℝ   -- Acceleration due to gravity

/-- The condition for the marble to return to the edge of the table -/
def returns_to_edge (m : MarbleMotion) : Prop :=
  m.v₀ = 2 * m.D * Real.sqrt (m.g / (2 * m.h))

/-- Theorem stating the condition for the marble to return to the edge of the table -/
theorem marble_return_condition (m : MarbleMotion) :
  returns_to_edge m ↔ m.v₀ = 2 * m.D * Real.sqrt (m.g / (2 * m.h)) :=
by sorry

end marble_return_condition_l755_75526


namespace ellipse_intersection_slope_product_l755_75551

/-- Given a line l passing through M(-2,0) with slope k1 (k1 ≠ 0) intersecting 
    the ellipse x^2 + 2y^2 = 4 at P1 and P2, and P being the midpoint of P1P2,
    if k2 is the slope of OP, then k1k2 = -1/2 -/
theorem ellipse_intersection_slope_product 
  (k1 : ℝ) (P1 P2 P : ℝ × ℝ) (k2 : ℝ)
  (h1 : k1 ≠ 0)
  (h2 : P1.1^2 + 2*P1.2^2 = 4)
  (h3 : P2.1^2 + 2*P2.2^2 = 4)
  (h4 : P1.2 = k1 * (P1.1 + 2))
  (h5 : P2.2 = k1 * (P2.1 + 2))
  (h6 : P = ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2))
  (h7 : k2 = P.2 / P.1) :
  k1 * k2 = -1/2 := by
sorry

end ellipse_intersection_slope_product_l755_75551


namespace duck_park_problem_l755_75523

theorem duck_park_problem (initial_ducks : ℕ) (geese_arrive : ℕ) (geese_leave : ℕ) : 
  initial_ducks = 25 →
  geese_arrive = 4 →
  geese_leave = 10 →
  ((2 * initial_ducks) - 10) - geese_leave - (initial_ducks + geese_arrive) = 1 := by
  sorry

end duck_park_problem_l755_75523


namespace max_value_proof_l755_75577

theorem max_value_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (abcd : ℝ) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) ≤ 1 :=
sorry

end max_value_proof_l755_75577


namespace simplify_sqrt_expression_l755_75541

theorem simplify_sqrt_expression :
  Real.sqrt 8 - Real.sqrt 50 + Real.sqrt 72 = 3 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_expression_l755_75541


namespace election_vote_count_l755_75544

theorem election_vote_count : 
  -- Define the total number of votes
  ∀ V : ℕ,
  -- First round vote percentages
  let a1 := (27 : ℚ) / 100 * V
  let b1 := (24 : ℚ) / 100 * V
  let c1 := (20 : ℚ) / 100 * V
  let d1 := (18 : ℚ) / 100 * V
  let e1 := V - (a1 + b1 + c1 + d1)
  -- Second round vote percentages
  let a2 := (30 : ℚ) / 100 * V
  let b2 := (27 : ℚ) / 100 * V
  let c2 := (22 : ℚ) / 100 * V
  let d2 := V - (a2 + b2 + c2)
  -- Final round
  let additional_votes := (10 : ℚ) / 100 * V  -- 5% each from C and D supporters
  let a_final := a2 + (5 : ℚ) / 100 * V
  let b_final := b2 + d2 + (5 : ℚ) / 100 * V
  -- B wins by 1350 votes
  b_final - a_final = 1350 →
  V = 7500 := by
sorry

end election_vote_count_l755_75544


namespace tea_sale_prices_l755_75567

/-- Calculates the sale price per kg for a given tea type -/
def salePricePerKg (quantity : ℕ) (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  (quantity * costPrice + quantity * costPrice * profitPercentage) / quantity

theorem tea_sale_prices :
  let teaA := salePricePerKg 80 15 (25/100)
  let teaB := salePricePerKg 20 20 (30/100)
  let teaC := salePricePerKg 50 25 (20/100)
  let teaD := salePricePerKg 30 30 (15/100)
  teaA = 75/4 ∧ teaB = 26 ∧ teaC = 30 ∧ teaD = 69/2 :=
by sorry

end tea_sale_prices_l755_75567


namespace negation_of_even_sum_false_l755_75505

theorem negation_of_even_sum_false : 
  ¬(∀ a b : ℤ, (¬(Even a ∧ Even b) → ¬Even (a + b))) := by sorry

end negation_of_even_sum_false_l755_75505


namespace profit_7500_at_65_max_profit_at_70_max_profit_is_8000_l755_75520

/-- Represents the online store's pricing and sales model -/
structure Store where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ

/-- Calculates the number of items sold based on the current price -/
def items_sold (s : Store) (price : ℝ) : ℝ :=
  s.initial_sales + s.price_sensitivity * (s.initial_price - price)

/-- Calculates the weekly profit based on the current price -/
def weekly_profit (s : Store) (price : ℝ) : ℝ :=
  (price - s.cost_price) * (items_sold s price)

/-- The store's pricing and sales model -/
def children_clothing_store : Store :=
  { cost_price := 50
  , initial_price := 80
  , initial_sales := 200
  , price_sensitivity := 20 }

/-- Theorem: The selling price of 65 yuan achieves a weekly profit of 7500 yuan while maximizing customer benefits -/
theorem profit_7500_at_65 :
  weekly_profit children_clothing_store 65 = 7500 ∧
  ∀ p, p < 65 → weekly_profit children_clothing_store p < 7500 :=
sorry

/-- Theorem: The selling price of 70 yuan maximizes the weekly profit -/
theorem max_profit_at_70 :
  ∀ p, weekly_profit children_clothing_store p ≤ weekly_profit children_clothing_store 70 :=
sorry

/-- Theorem: The maximum weekly profit is 8000 yuan -/
theorem max_profit_is_8000 :
  weekly_profit children_clothing_store 70 = 8000 :=
sorry

end profit_7500_at_65_max_profit_at_70_max_profit_is_8000_l755_75520


namespace starters_with_twin_restriction_l755_75536

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of players in the team -/
def total_players : ℕ := 16

/-- The number of starters to be chosen -/
def starters : ℕ := 5

/-- The number of players excluding both sets of twins -/
def players_excluding_twins : ℕ := total_players - 4

/-- The number of ways to choose starters with the twin restriction -/
def ways_to_choose_starters : ℕ :=
  binomial total_players starters -
  2 * binomial (total_players - 2) (starters - 2) +
  binomial (total_players - 4) (starters - 4)

theorem starters_with_twin_restriction :
  ways_to_choose_starters = 3652 :=
sorry

end starters_with_twin_restriction_l755_75536


namespace sean_sunday_spending_l755_75573

/-- Represents Sean's Sunday purchases and their costs --/
structure SundayPurchases where
  almond_croissant_price : ℝ
  salami_cheese_croissant_price : ℝ
  plain_croissant_price : ℝ
  focaccia_price : ℝ
  latte_price : ℝ
  almond_croissant_quantity : ℕ
  salami_cheese_croissant_quantity : ℕ
  plain_croissant_quantity : ℕ
  focaccia_quantity : ℕ
  latte_quantity : ℕ

/-- Calculates the total cost of Sean's Sunday purchases --/
def total_cost (purchases : SundayPurchases) : ℝ :=
  purchases.almond_croissant_price * purchases.almond_croissant_quantity +
  purchases.salami_cheese_croissant_price * purchases.salami_cheese_croissant_quantity +
  purchases.plain_croissant_price * purchases.plain_croissant_quantity +
  purchases.focaccia_price * purchases.focaccia_quantity +
  purchases.latte_price * purchases.latte_quantity

/-- Theorem stating that Sean's total spending on Sunday is $21.00 --/
theorem sean_sunday_spending (purchases : SundayPurchases)
  (h1 : purchases.almond_croissant_price = 4.5)
  (h2 : purchases.salami_cheese_croissant_price = 4.5)
  (h3 : purchases.plain_croissant_price = 3)
  (h4 : purchases.focaccia_price = 4)
  (h5 : purchases.latte_price = 2.5)
  (h6 : purchases.almond_croissant_quantity = 1)
  (h7 : purchases.salami_cheese_croissant_quantity = 1)
  (h8 : purchases.plain_croissant_quantity = 1)
  (h9 : purchases.focaccia_quantity = 1)
  (h10 : purchases.latte_quantity = 2)
  : total_cost purchases = 21 := by
  sorry

end sean_sunday_spending_l755_75573


namespace g_composition_result_l755_75539

noncomputable def g (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^3 else -z^3

theorem g_composition_result :
  g (g (g (g (1 + I)))) = -134217728 - 134217728 * I :=
by sorry

end g_composition_result_l755_75539


namespace solve_trading_card_problem_l755_75564

def trading_card_problem (initial_cards : ℕ) (brother_sets : ℕ) (friend_sets : ℕ) 
  (total_given : ℕ) (cards_per_set : ℕ) : ℕ :=
  let cards_to_brother := brother_sets * cards_per_set
  let cards_to_friend := friend_sets * cards_per_set
  let remaining_cards := total_given - (cards_to_brother + cards_to_friend)
  remaining_cards / cards_per_set

theorem solve_trading_card_problem :
  trading_card_problem 365 8 2 195 13 = 5 := by
  sorry

end solve_trading_card_problem_l755_75564


namespace positive_number_square_root_l755_75546

theorem positive_number_square_root (x : ℝ) : 
  x > 0 → (Real.sqrt ((4 * x) / 3) = x) → x = 4 / 3 := by
  sorry

end positive_number_square_root_l755_75546


namespace rabbits_total_distance_l755_75572

/-- The total distance hopped by two rabbits in a given time -/
def total_distance (white_speed brown_speed time : ℕ) : ℕ :=
  (white_speed * time) + (brown_speed * time)

/-- Theorem: The total distance hopped by the white and brown rabbits in 5 minutes is 135 meters -/
theorem rabbits_total_distance :
  total_distance 15 12 5 = 135 := by
  sorry

end rabbits_total_distance_l755_75572


namespace volunteer_assignment_problem_l755_75555

/-- The number of ways to assign n volunteers to k venues with at least one volunteer at each venue -/
def assignment_count (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

/-- The problem statement -/
theorem volunteer_assignment_problem :
  assignment_count 5 3 = 150 := by
  sorry

end volunteer_assignment_problem_l755_75555
