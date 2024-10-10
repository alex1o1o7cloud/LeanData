import Mathlib

namespace geometric_sequence_sum_l1495_149559

theorem geometric_sequence_sum : 
  let a : ℚ := 1/3  -- first term
  let r : ℚ := 1/3  -- common ratio
  let n : ℕ := 5    -- number of terms
  let S : ℚ := (a * (1 - r^n)) / (1 - r)  -- sum formula
  S = 121/243 := by
sorry

end geometric_sequence_sum_l1495_149559


namespace extrema_of_f_l1495_149594

def f (x : ℝ) := -x^2 + x + 1

theorem extrema_of_f :
  let a := 0
  let b := 3/2
  ∃ (x_min x_max : ℝ), x_min ∈ Set.Icc a b ∧ x_max ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    f x_min = 1/4 ∧ f x_max = 5/4 :=
by sorry

end extrema_of_f_l1495_149594


namespace equation_roots_l1495_149535

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => -x * (x + 3) - x * (x + 3)
  (f 0 = 0 ∧ f (-3) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 0 ∨ x = -3) :=
by sorry

end equation_roots_l1495_149535


namespace shaded_fraction_of_square_l1495_149517

/-- Given a square with side length x, where P is at a corner and Q is at the midpoint of an adjacent side,
    the fraction of the square's interior that is shaded is 3/4. -/
theorem shaded_fraction_of_square (x : ℝ) (h : x > 0) : 
  let square_area := x^2
  let triangle_area := (1/2) * x * (x/2)
  let shaded_area := square_area - triangle_area
  shaded_area / square_area = 3/4 := by
sorry

end shaded_fraction_of_square_l1495_149517


namespace smallest_integer_satisfying_inequality_l1495_149561

theorem smallest_integer_satisfying_inequality :
  ∃ (y : ℤ), (7 - 3 * y ≤ 29) ∧ (∀ (z : ℤ), z < y → 7 - 3 * z > 29) ∧ y = -7 := by
  sorry

end smallest_integer_satisfying_inequality_l1495_149561


namespace max_value_on_parabola_l1495_149592

/-- The maximum value of m + n for a point (m, n) on the graph of y = -x^2 + 3 is 13/4 -/
theorem max_value_on_parabola : 
  ∃ (max : ℝ), max = 13/4 ∧ 
  ∀ (m n : ℝ), n = -m^2 + 3 → m + n ≤ max :=
sorry

end max_value_on_parabola_l1495_149592


namespace haley_trees_after_typhoon_l1495_149553

/-- The number of trees Haley has left after a typhoon -/
def trees_left (initial_trees dead_trees : ℕ) : ℕ :=
  initial_trees - dead_trees

/-- Theorem: Haley has 10 trees left after the typhoon -/
theorem haley_trees_after_typhoon :
  trees_left 12 2 = 10 := by
  sorry

end haley_trees_after_typhoon_l1495_149553


namespace quadratic_minimum_value_l1495_149518

def f (x : ℝ) : ℝ := 2 * x^2 + 6 * x + 5

theorem quadratic_minimum_value :
  (f 1 = 13) →
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 0.5 :=
by sorry

end quadratic_minimum_value_l1495_149518


namespace company_uses_systematic_sampling_l1495_149595

/-- Represents a sampling method -/
inductive SamplingMethod
| LotteryMethod
| RandomNumberTableMethod
| SystematicSampling
| StratifiedSampling

/-- Represents a production line -/
structure ProductionLine :=
  (uniform : Bool)

/-- Represents a sampling process -/
structure SamplingProcess :=
  (line : ProductionLine)
  (interval : ℕ)

/-- Determines if a sampling process is systematic -/
def is_systematic (process : SamplingProcess) : Prop :=
  process.line.uniform ∧ process.interval > 0

/-- The company's sampling method -/
def company_sampling : SamplingProcess :=
  { line := { uniform := true },
    interval := 10 }

/-- Theorem stating that the company's sampling method is systematic sampling -/
theorem company_uses_systematic_sampling :
  is_systematic company_sampling ∧ 
  SamplingMethod.SystematicSampling = 
    (match company_sampling with
     | { line := { uniform := true }, interval := 10 } => SamplingMethod.SystematicSampling
     | _ => SamplingMethod.LotteryMethod) :=
sorry

end company_uses_systematic_sampling_l1495_149595


namespace min_value_theorem_l1495_149564

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 2) :
  2 / m + 1 / n ≥ 4 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 2 ∧ 2 / m₀ + 1 / n₀ = 4 :=
by sorry

end min_value_theorem_l1495_149564


namespace system_solution_implies_m_value_l1495_149528

theorem system_solution_implies_m_value (x y m : ℝ) : 
  (2 * x + y = 6 * m) →
  (3 * x - 2 * y = 2 * m) →
  (x / 3 - y / 5 = 4) →
  m = 15 := by
sorry

end system_solution_implies_m_value_l1495_149528


namespace triangle_area_l1495_149509

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : 
  (1/2) * a * b = 24 := by
  sorry

end triangle_area_l1495_149509


namespace min_real_part_x_l1495_149504

theorem min_real_part_x (x y : ℂ) 
  (eq1 : x + 2 * y^2 = x^4)
  (eq2 : y + 2 * x^2 = y^4) :
  Real.sqrt (Real.sqrt ((1 - Real.sqrt 33) / 2)) ≤ x.re :=
sorry

end min_real_part_x_l1495_149504


namespace rightmost_three_digits_of_7_to_1987_l1495_149574

theorem rightmost_three_digits_of_7_to_1987 :
  7^1987 % 1000 = 543 := by
  sorry

end rightmost_three_digits_of_7_to_1987_l1495_149574


namespace euler_6_years_or_more_percentage_l1495_149597

/-- Represents the number of units for each tenure range in the bar graph --/
structure EmployeeDistribution where
  less_than_2_years : ℕ
  two_to_4_years : ℕ
  four_to_6_years : ℕ
  six_to_8_years : ℕ
  eight_to_10_years : ℕ
  more_than_10_years : ℕ

/-- Calculates the percentage of employees who have worked for 6 years or more --/
def percentage_6_years_or_more (d : EmployeeDistribution) : ℚ :=
  let total := d.less_than_2_years + d.two_to_4_years + d.four_to_6_years +
                d.six_to_8_years + d.eight_to_10_years + d.more_than_10_years
  let six_plus := d.six_to_8_years + d.eight_to_10_years + d.more_than_10_years
  (six_plus : ℚ) / (total : ℚ) * 100

/-- The actual distribution of employees at Euler Company --/
def euler_distribution : EmployeeDistribution :=
  { less_than_2_years := 4
  , two_to_4_years := 6
  , four_to_6_years := 7
  , six_to_8_years := 3
  , eight_to_10_years := 2
  , more_than_10_years := 1 }

theorem euler_6_years_or_more_percentage :
  percentage_6_years_or_more euler_distribution = 26 := by
  sorry

end euler_6_years_or_more_percentage_l1495_149597


namespace faster_runner_overtakes_in_five_laps_l1495_149583

/-- The length of the track in meters -/
def track_length : ℝ := 400

/-- The speed ratio of the faster runner to the slower runner -/
def speed_ratio : ℝ := 1.25

/-- The number of laps after which the faster runner overtakes the slower runner -/
def overtake_laps : ℝ := 5

/-- Theorem stating that the faster runner overtakes the slower runner after 5 laps -/
theorem faster_runner_overtakes_in_five_laps :
  ∀ (v : ℝ), v > 0 →
  speed_ratio * v * (overtake_laps * track_length / v) =
  (overtake_laps + 1) * track_length :=
by sorry

end faster_runner_overtakes_in_five_laps_l1495_149583


namespace remainder_of_M_divided_by_500_l1495_149565

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def product_of_factorials : ℕ := (List.range 50).foldl (λ acc i => acc * factorial (i + 1)) 1

def M : ℕ := (product_of_factorials.digits 10).reverse.takeWhile (·= 0) |>.length

theorem remainder_of_M_divided_by_500 : M % 500 = 391 := by
  sorry

end remainder_of_M_divided_by_500_l1495_149565


namespace bracket_mult_equation_solution_l1495_149503

-- Define the operation
def bracket_mult (a b c d : ℝ) : ℝ := a * c - b * d

-- State the theorem
theorem bracket_mult_equation_solution :
  ∃ (x : ℝ), (bracket_mult (-x) 3 (x - 2) (-6) = 10) ∧ (x = 4 ∨ x = -2) :=
by sorry

end bracket_mult_equation_solution_l1495_149503


namespace change_in_average_weight_l1495_149551

/-- The change in average weight when replacing a person in a group -/
theorem change_in_average_weight 
  (n : ℕ) 
  (old_weight new_weight : ℝ) 
  (h1 : n = 6) 
  (h2 : old_weight = 69) 
  (h3 : new_weight = 79.8) : 
  (new_weight - old_weight) / n = 1.8 := by
  sorry

end change_in_average_weight_l1495_149551


namespace perfect_square_polynomial_l1495_149538

/-- If x^2 - kx + 25 is a perfect square polynomial, then k = ±10 -/
theorem perfect_square_polynomial (k : ℝ) : 
  (∃ (a : ℝ), ∀ x, x^2 - k*x + 25 = (x - a)^2) → (k = 10 ∨ k = -10) := by
  sorry

end perfect_square_polynomial_l1495_149538


namespace binomial_expansion_problem_l1495_149588

theorem binomial_expansion_problem (a b : ℝ) (n : ℕ) :
  (∀ k, 1 ≤ k ∧ k ≤ n + 1 → Nat.choose n (k - 1) ≤ Nat.choose n 5) ∧
  (a + b = 4) →
  (n = 10) ∧
  ((4^n + 7) % 3 = 2) := by
sorry

end binomial_expansion_problem_l1495_149588


namespace max_elements_in_S_l1495_149520

theorem max_elements_in_S (A : Finset ℝ) (h_card : A.card = 100) (h_pos : ∀ a ∈ A, a > 0) :
  let S := {p : ℝ × ℝ | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 - p.2 ∈ A}
  (Finset.filter (fun p => p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 - p.2 ∈ A) (A.product A)).card ≤ 4950 :=
by sorry

end max_elements_in_S_l1495_149520


namespace hash_problem_l1495_149589

-- Define the # operation
def hash (a b : ℚ) : ℚ := a + (a^2 / b)

-- Theorem statement
theorem hash_problem : (hash 4 3) - 10 = -2/3 := by
  sorry

end hash_problem_l1495_149589


namespace exponential_graph_quadrants_l1495_149523

theorem exponential_graph_quadrants (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : b < -1) :
  ∀ x y : ℝ, y = a^x + b → ¬(x > 0 ∧ y > 0) :=
by sorry

end exponential_graph_quadrants_l1495_149523


namespace total_numbers_correct_l1495_149516

/-- Represents a student in the talent show -/
inductive Student : Type
| Sarah : Student
| Ben : Student
| Jake : Student
| Lily : Student

/-- The total number of musical numbers in the show -/
def total_numbers : ℕ := 7

/-- The number of songs Sarah performed -/
def sarah_songs : ℕ := 6

/-- The number of songs Ben performed -/
def ben_songs : ℕ := sarah_songs - 3

/-- The number of songs Jake performed -/
def jake_songs : ℕ := 6

/-- The number of songs Lily performed -/
def lily_songs : ℕ := 6

/-- The number of duo shows Jake and Lily performed together -/
def jake_lily_duo : ℕ := 1

/-- The number of shows Jake and Lily performed together -/
def jake_lily_together : ℕ := 6

/-- Theorem stating that the total number of musical numbers is correct -/
theorem total_numbers_correct : 
  (sarah_songs = total_numbers - 2) ∧ 
  (ben_songs = sarah_songs - 3) ∧
  (jake_songs = lily_songs) ∧
  (jake_lily_together ≤ 7) ∧
  (jake_lily_together > jake_songs - jake_lily_duo) ∧
  (total_numbers = jake_songs + 1) := by
  sorry

#check total_numbers_correct

end total_numbers_correct_l1495_149516


namespace rotation_and_scaling_l1495_149527

def rotate90Clockwise (z : ℂ) : ℂ := -z.im + z.re * Complex.I

theorem rotation_and_scaling :
  let z : ℂ := 3 + 4 * Complex.I
  let rotated := rotate90Clockwise z
  let scaled := 2 * rotated
  scaled = -8 - 6 * Complex.I := by sorry

end rotation_and_scaling_l1495_149527


namespace number_of_divisors_sum_of_divisors_l1495_149566

def n : ℕ := 2310

-- Function to count positive divisors
def count_divisors (m : ℕ) : ℕ := sorry

-- Function to sum positive divisors
def sum_divisors (m : ℕ) : ℕ := sorry

-- Theorem stating the number of positive divisors of 2310
theorem number_of_divisors : count_divisors n = 32 := by sorry

-- Theorem stating the sum of positive divisors of 2310
theorem sum_of_divisors : sum_divisors n = 6912 := by sorry

end number_of_divisors_sum_of_divisors_l1495_149566


namespace no_m_exists_for_all_x_inequality_l1495_149513

theorem no_m_exists_for_all_x_inequality :
  ¬ ∃ m : ℝ, ∀ x : ℝ, m * x^2 - 2*x - m + 1 < 0 := by
  sorry

end no_m_exists_for_all_x_inequality_l1495_149513


namespace concert_ticket_revenue_l1495_149573

/-- Calculates the total revenue from concert ticket sales --/
theorem concert_ticket_revenue : 
  let ticket_price : ℕ := 20
  let first_discount : ℕ := 40
  let second_discount : ℕ := 15
  let first_group : ℕ := 10
  let second_group : ℕ := 20
  let total_tickets : ℕ := 50
  
  let first_group_revenue := first_group * (ticket_price - (ticket_price * first_discount / 100))
  let second_group_revenue := second_group * (ticket_price - (ticket_price * second_discount / 100))
  let full_price_revenue := (total_tickets - first_group - second_group) * ticket_price
  
  first_group_revenue + second_group_revenue + full_price_revenue = 860 :=
by
  sorry


end concert_ticket_revenue_l1495_149573


namespace existence_of_mn_l1495_149546

theorem existence_of_mn (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ m n : ℕ, m + n < p ∧ p ∣ (2^m * 3^n - 1) := by
  sorry

end existence_of_mn_l1495_149546


namespace fraction_equality_l1495_149542

theorem fraction_equality : (1/4 - 1/5) / (1/3 - 1/6 + 1/12) = 1/5 := by
  sorry

end fraction_equality_l1495_149542


namespace turnip_bag_weights_l1495_149586

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (onion_weight carrot_weight : ℕ),
    onion_weight + carrot_weight = (bag_weights.sum - t) ∧
    carrot_weight = 2 * onion_weight ∧
    ∃ (onion_bags carrot_bags : List ℕ),
      onion_bags ++ carrot_bags = bag_weights.filter (λ x => x ≠ t) ∧
      onion_bags.sum = onion_weight ∧
      carrot_bags.sum = carrot_weight

theorem turnip_bag_weights :
  {t : ℕ | is_valid_turnip_weight t} = {13, 16} := by
  sorry

end turnip_bag_weights_l1495_149586


namespace thirteenth_square_vs_first_twelve_l1495_149556

def grains (k : ℕ) : ℕ := 2^k

def sum_grains (n : ℕ) : ℕ := (grains (n + 1)) - 2

theorem thirteenth_square_vs_first_twelve :
  grains 13 = sum_grains 12 + 2 := by
  sorry

end thirteenth_square_vs_first_twelve_l1495_149556


namespace f_is_odd_l1495_149508

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + Real.sin x) / Real.cos x)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_is_odd : is_odd f := by sorry

end f_is_odd_l1495_149508


namespace books_and_students_count_l1495_149591

/-- The number of books distributed to students -/
def total_books : ℕ := 26

/-- The number of students receiving books -/
def total_students : ℕ := 6

/-- Condition 1: If each person receives 3 books, there will be 8 books left -/
axiom condition1 : total_books = 3 * total_students + 8

/-- Condition 2: If each of the previous students receives 5 books, 
    then the last person will not receive 2 books -/
axiom condition2 : 
  total_books - 5 * (total_students - 1) < 2 ∧ 
  total_books - 5 * (total_students - 1) ≥ 0

/-- Theorem: Given the conditions, prove that the number of books is 26 
    and the number of students is 6 -/
theorem books_and_students_count : 
  total_books = 26 ∧ total_students = 6 := by
  sorry

end books_and_students_count_l1495_149591


namespace pants_purchase_l1495_149531

theorem pants_purchase (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (total_paid : ℝ) :
  original_price = 45 →
  discount_rate = 0.20 →
  tax_rate = 0.10 →
  total_paid = 396 →
  ∃ (num_pairs : ℕ), 
    (num_pairs : ℝ) * (original_price * (1 - discount_rate) * (1 + tax_rate)) = total_paid ∧
    num_pairs = 10 := by
  sorry

end pants_purchase_l1495_149531


namespace large_birdhouses_sold_l1495_149515

/-- Represents the number of large birdhouses sold -/
def large_birdhouses : ℕ := sorry

/-- The price of a large birdhouse in dollars -/
def large_price : ℕ := 22

/-- The price of a medium birdhouse in dollars -/
def medium_price : ℕ := 16

/-- The price of a small birdhouse in dollars -/
def small_price : ℕ := 7

/-- The number of medium birdhouses sold -/
def medium_sold : ℕ := 2

/-- The number of small birdhouses sold -/
def small_sold : ℕ := 3

/-- The total sales in dollars -/
def total_sales : ℕ := 97

/-- Theorem stating that the number of large birdhouses sold is 2 -/
theorem large_birdhouses_sold : large_birdhouses = 2 := by
  sorry

end large_birdhouses_sold_l1495_149515


namespace min_bilingual_students_l1495_149558

theorem min_bilingual_students (total : ℕ) (hindi : ℕ) (english : ℕ) 
  (h_total : total = 40)
  (h_hindi : hindi = 30)
  (h_english : english = 20) :
  ∃ (both : ℕ), both ≥ hindi + english - total ∧ 
  (∀ (x : ℕ), x ≥ hindi + english - total → x ≥ both) :=
by sorry

end min_bilingual_students_l1495_149558


namespace sum_proper_divisors_256_l1495_149568

theorem sum_proper_divisors_256 : 
  (Finset.filter (fun d => d ≠ 256 ∧ 256 % d = 0) (Finset.range 257)).sum id = 255 := by
  sorry

end sum_proper_divisors_256_l1495_149568


namespace existence_condition_l1495_149512

theorem existence_condition (a : ℝ) : 
  (∃ x y : ℝ, Real.sqrt (2 * x * y + a) = x + y + 17) ↔ a ≥ -289/2 := by
sorry

end existence_condition_l1495_149512


namespace eighth_term_is_sixteen_l1495_149593

def odd_term (n : ℕ) : ℕ := 2 * n - 1

def even_term (n : ℕ) : ℕ := 4 * n

def sequence_term (n : ℕ) : ℕ :=
  if n % 2 = 1 then odd_term ((n + 1) / 2) else even_term (n / 2)

theorem eighth_term_is_sixteen : sequence_term 8 = 16 := by
  sorry

end eighth_term_is_sixteen_l1495_149593


namespace expand_product_l1495_149563

theorem expand_product (x : ℝ) : (x + 3) * (x - 4) = x^2 - x - 12 := by
  sorry

end expand_product_l1495_149563


namespace nathan_ate_100_gumballs_l1495_149530

/-- The number of gumballs in each package -/
def gumballs_per_package : ℝ := 5.0

/-- The number of packages Nathan ate -/
def packages_eaten : ℝ := 20.0

/-- The total number of gumballs Nathan ate -/
def total_gumballs : ℝ := gumballs_per_package * packages_eaten

theorem nathan_ate_100_gumballs : total_gumballs = 100.0 := by
  sorry

end nathan_ate_100_gumballs_l1495_149530


namespace worker_completion_time_l1495_149584

theorem worker_completion_time (worker_b_time worker_ab_time : Real) 
  (hb : worker_b_time = 10)
  (hab : worker_ab_time = 3.333333333333333)
  : ∃ worker_a_time : Real, 
    worker_a_time = 5 ∧ 
    1 / worker_a_time + 1 / worker_b_time = 1 / worker_ab_time :=
by
  sorry

end worker_completion_time_l1495_149584


namespace endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l1495_149571

/-- Given a line segment with one endpoint (5,4) and midpoint (3.5,10.5),
    the sum of the coordinates of the other endpoint is 19. -/
theorem endpoint_coordinate_sum : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun x₁ y₁ x_mid y_mid x₂ y₂ =>
    x₁ = 5 ∧ y₁ = 4 ∧ x_mid = 3.5 ∧ y_mid = 10.5 ∧
    x_mid = (x₁ + x₂) / 2 ∧ y_mid = (y₁ + y₂) / 2 →
    x₂ + y₂ = 19

/-- Proof of the theorem -/
theorem endpoint_coordinate_sum_proof : 
  ∃ x₂ y₂, endpoint_coordinate_sum 5 4 3.5 10.5 x₂ y₂ := by
  sorry

end endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l1495_149571


namespace point_inside_ellipse_l1495_149587

theorem point_inside_ellipse (a : ℝ) : 
  (a^2 / 4 + 1 / 2 < 1) → (-Real.sqrt 2 < a ∧ a < Real.sqrt 2) := by
  sorry

end point_inside_ellipse_l1495_149587


namespace water_in_bucket_l1495_149534

theorem water_in_bucket (initial_water : ℝ) (poured_out : ℝ) (remaining_water : ℝ) :
  initial_water = 0.8 →
  poured_out = 0.2 →
  remaining_water = initial_water - poured_out →
  remaining_water = 0.6 := by
sorry

end water_in_bucket_l1495_149534


namespace sequence_properties_l1495_149562

/-- Sequence b_n with sum of first n terms S_n -/
def b : ℕ → ℝ := sorry

/-- Sum of first n terms of b_n -/
def S : ℕ → ℝ := sorry

/-- Arithmetic sequence c_n -/
def c : ℕ → ℝ := sorry

/-- Sequence a_n formed by common terms of b_n and c_n in ascending order -/
def a : ℕ → ℝ := sorry

/-- The product of the first n terms of a_n -/
def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, 2 * S n = 3 * (b n - 1)) ∧ 
  (c 1 = 5) ∧
  (c 1 + c 2 + c 3 = 27) →
  (∀ n : ℕ, b n = 3^n) ∧
  (∀ n : ℕ, c n = 4*n + 1) ∧
  (T 20 = 9^210) := by sorry

end sequence_properties_l1495_149562


namespace star_op_equivalence_l1495_149507

-- Define the ※ operation
def star_op (m n : ℝ) : ℝ := m * n - m - n + 3

-- State the theorem
theorem star_op_equivalence (x : ℝ) :
  6 < star_op 2 x ∧ star_op 2 x < 7 ↔ 5 < x ∧ x < 6 := by
  sorry

end star_op_equivalence_l1495_149507


namespace race_car_time_problem_l1495_149536

theorem race_car_time_problem (time_A time_sync : ℕ) (time_B : ℕ) : 
  time_A = 28 →
  time_sync = 168 →
  time_sync % time_A = 0 →
  time_sync % time_B = 0 →
  time_B > time_A →
  time_B < time_sync →
  (time_sync / time_A) % (time_sync / time_B) = 0 →
  time_B = 42 :=
by sorry

end race_car_time_problem_l1495_149536


namespace congruence_problem_l1495_149525

theorem congruence_problem (x : ℤ) :
  (4 * x + 9) % 17 = 3 → (3 * x + 12) % 17 = 16 := by
  sorry

end congruence_problem_l1495_149525


namespace geometric_sequence_sum_l1495_149554

theorem geometric_sequence_sum (a : ℕ → ℚ) (q : ℚ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- a is a geometric sequence with common ratio q
  a 1 + a 2 + a 3 + a 4 = 15/8 →    -- sum of first four terms
  a 2 * a 3 = -9/8 →                -- product of second and third terms
  1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 = -5/3 := by
sorry

end geometric_sequence_sum_l1495_149554


namespace cindy_envelopes_left_l1495_149514

def envelopes_left (initial_envelopes : ℕ) (num_friends : ℕ) (envelopes_per_friend : ℕ) : ℕ :=
  initial_envelopes - (num_friends * envelopes_per_friend)

theorem cindy_envelopes_left :
  let initial_envelopes : ℕ := 74
  let num_friends : ℕ := 11
  let envelopes_per_friend : ℕ := 6
  envelopes_left initial_envelopes num_friends envelopes_per_friend = 8 := by
  sorry

end cindy_envelopes_left_l1495_149514


namespace total_tickets_is_340_l1495_149576

/-- Represents the number of tickets sold for a theater performance. -/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- The total revenue from ticket sales. -/
def totalRevenue (sales : TicketSales) : ℕ :=
  12 * sales.orchestra + 8 * sales.balcony

/-- The difference between balcony and orchestra ticket sales. -/
def balconyOrchestraDiff (sales : TicketSales) : ℤ :=
  sales.balcony - sales.orchestra

/-- The total number of tickets sold. -/
def totalTickets (sales : TicketSales) : ℕ :=
  sales.orchestra + sales.balcony

/-- Theorem stating that given the conditions, the total number of tickets sold is 340. -/
theorem total_tickets_is_340 :
  ∃ (sales : TicketSales),
    totalRevenue sales = 3320 ∧
    balconyOrchestraDiff sales = 40 ∧
    totalTickets sales = 340 := by
  sorry


end total_tickets_is_340_l1495_149576


namespace unique_ages_l1495_149543

def is_valid_ages (f : ℤ → ℤ) (b c : ℤ) : Prop :=
  (∀ x y : ℤ, x - y ∣ f x - f y) ∧
  f 7 = 77 ∧
  f b = 85 ∧
  f c = 0 ∧
  7 < b ∧
  b < c

theorem unique_ages :
  ∀ f b c, is_valid_ages f b c → b = 9 ∧ c = 14 := by
sorry

end unique_ages_l1495_149543


namespace bookshop_inventory_l1495_149532

/-- The initial number of books in John's bookshop -/
def initial_books : ℕ := 900

/-- The number of books sold on Monday -/
def monday_sales : ℕ := 75

/-- The number of books sold on Tuesday -/
def tuesday_sales : ℕ := 50

/-- The number of books sold on Wednesday -/
def wednesday_sales : ℕ := 64

/-- The number of books sold on Thursday -/
def thursday_sales : ℕ := 78

/-- The number of books sold on Friday -/
def friday_sales : ℕ := 135

/-- The percentage of books that were not sold -/
def unsold_percentage : ℚ := 55333333333333336 / 100000000000000000

theorem bookshop_inventory :
  initial_books = 900 ∧
  (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales : ℚ) / initial_books = 1 - unsold_percentage :=
by sorry

end bookshop_inventory_l1495_149532


namespace A_and_B_complementary_l1495_149549

-- Define the sample space for a die toss
def DieOutcome := Fin 6

-- Define events A, B, and C
def eventA (outcome : DieOutcome) : Prop := outcome.val ≤ 3
def eventB (outcome : DieOutcome) : Prop := outcome.val ≥ 4
def eventC (outcome : DieOutcome) : Prop := outcome.val % 2 = 1

-- Theorem stating that A and B are complementary events
theorem A_and_B_complementary :
  ∀ (outcome : DieOutcome), eventA outcome ↔ ¬ eventB outcome :=
by sorry

end A_and_B_complementary_l1495_149549


namespace seth_purchase_difference_l1495_149502

/-- Calculates the difference in cost between discounted ice cream and yogurt purchases. -/
def ice_cream_yogurt_cost_difference (
  ice_cream_cartons : ℕ)
  (yogurt_cartons : ℕ)
  (ice_cream_price : ℚ)
  (yogurt_price : ℚ)
  (ice_cream_discount : ℚ)
  (yogurt_discount : ℚ) : ℚ :=
  let ice_cream_cost := ice_cream_cartons * ice_cream_price * (1 - ice_cream_discount)
  let yogurt_cost := yogurt_cartons * yogurt_price * (1 - yogurt_discount)
  ice_cream_cost - yogurt_cost

theorem seth_purchase_difference :
  ice_cream_yogurt_cost_difference 20 2 6 1 (1/10) (1/5) = 1064/10 := by
  sorry

end seth_purchase_difference_l1495_149502


namespace oil_barrel_ratio_l1495_149582

theorem oil_barrel_ratio (mass_A mass_B : ℝ) : 
  (mass_A + 10000 : ℝ) / (mass_B + 10000) = 4 / 5 →
  (mass_A + 18000 : ℝ) / (mass_B + 2000) = 8 / 7 →
  mass_A / mass_B = 3 / 4 :=
by sorry

end oil_barrel_ratio_l1495_149582


namespace star_equation_solution_l1495_149501

-- Define the star operation
def star (a b : ℚ) : ℚ := a * b + 3 * b - a

-- State the theorem
theorem star_equation_solution :
  ∀ y : ℚ, star 4 y = 40 → y = 44 / 7 := by
  sorry

end star_equation_solution_l1495_149501


namespace negation_of_proposition_l1495_149560

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end negation_of_proposition_l1495_149560


namespace ellipse_properties_l1495_149579

-- Define the ellipse C
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y m : ℝ) : Prop :=
  y = x + m

-- Define the theorem
theorem ellipse_properties
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (h_triangle : ∃ A F₁ F₂ : ℝ × ℝ, 
    ellipse A.1 A.2 a b ∧
    ellipse F₁.1 F₁.2 a b ∧
    ellipse F₂.1 F₂.2 a b ∧
    (A.2 - F₁.2)^2 + (A.1 - F₁.1)^2 = (A.2 - F₂.2)^2 + (A.1 - F₂.1)^2 ∧
    (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2 = 8) :
  (∀ x y, ellipse x y 2 (Real.sqrt 2)) ∧
  (∀ P Q : ℝ × ℝ, 
    ellipse P.1 P.2 2 (Real.sqrt 2) ∧
    ellipse Q.1 Q.2 2 (Real.sqrt 2) ∧
    line P.1 P.2 1 ∧
    line Q.1 Q.2 1 →
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 80/9) ∧
  (∀ m : ℝ, 
    (∃ P Q : ℝ × ℝ,
      ellipse P.1 P.2 2 (Real.sqrt 2) ∧
      ellipse Q.1 Q.2 2 (Real.sqrt 2) ∧
      line P.1 P.2 m ∧
      line Q.1 Q.2 m ∧
      P.1 * Q.2 - P.2 * Q.1 = 8/3) ↔
    (m = 2 ∨ m = -2 ∨ m = Real.sqrt 2 ∨ m = -Real.sqrt 2)) :=
by sorry

end ellipse_properties_l1495_149579


namespace total_fish_in_lake_l1495_149524

/-- The number of fish per white duck -/
def fishPerWhiteDuck : ℕ := 5

/-- The number of fish per black duck -/
def fishPerBlackDuck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fishPerMulticolorDuck : ℕ := 12

/-- The number of white ducks -/
def numWhiteDucks : ℕ := 3

/-- The number of black ducks -/
def numBlackDucks : ℕ := 7

/-- The number of multicolor ducks -/
def numMulticolorDucks : ℕ := 6

/-- The total number of fish in the lake -/
def totalFish : ℕ := fishPerWhiteDuck * numWhiteDucks + 
                     fishPerBlackDuck * numBlackDucks + 
                     fishPerMulticolorDuck * numMulticolorDucks

theorem total_fish_in_lake : totalFish = 157 := by
  sorry

end total_fish_in_lake_l1495_149524


namespace stamp_costs_l1495_149575

theorem stamp_costs (a b c d : ℝ) : 
  a + b + c + d = 84 →                   -- sum is 84
  b - a = c - b ∧ c - b = d - c →        -- arithmetic progression
  d = 2.5 * a →                          -- largest is 2.5 times smallest
  a = 12 ∧ b = 18 ∧ c = 24 ∧ d = 30 :=   -- prove the values
by sorry

end stamp_costs_l1495_149575


namespace inequality_proof_l1495_149544

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d = 3) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 3/4) ∧
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) ≥ 2/3) :=
by sorry

end inequality_proof_l1495_149544


namespace not_all_new_releases_implies_exists_not_new_and_not_all_new_l1495_149500

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being a new release
variable (is_new_release : Book → Prop)

-- Define the library as a set of books
variable (library : Set Book)

-- Theorem stating that if not all books are new releases, 
-- then there exists a book that is not a new release and not all books are new releases
theorem not_all_new_releases_implies_exists_not_new_and_not_all_new
  (h : ¬(∀ b ∈ library, is_new_release b)) :
  (∃ b ∈ library, ¬(is_new_release b)) ∧ ¬(∀ b ∈ library, is_new_release b) := by
sorry

end not_all_new_releases_implies_exists_not_new_and_not_all_new_l1495_149500


namespace units_digit_of_2143_power_752_l1495_149577

theorem units_digit_of_2143_power_752 : (2143^752) % 10 = 1 := by
  sorry

end units_digit_of_2143_power_752_l1495_149577


namespace probability_triangle_from_random_chords_probability_is_favorable_over_total_total_pairings_calculation_favorable_pairings_is_one_probability_triangle_from_random_chords_value_l1495_149550

/-- The probability of forming a triangle when choosing three random chords on a circle -/
theorem probability_triangle_from_random_chords : ℚ :=
  1 / 15

/-- The number of ways to pair 6 points into three pairs -/
def total_pairings : ℕ := 15

/-- The number of pairings that result in all chords intersecting and forming a triangle -/
def favorable_pairings : ℕ := 1

theorem probability_is_favorable_over_total :
  probability_triangle_from_random_chords = favorable_pairings / total_pairings :=
sorry

theorem total_pairings_calculation :
  total_pairings = (1 / 6 : ℚ) * (Nat.choose 6 2) * (Nat.choose 4 2) * (Nat.choose 2 2) :=
sorry

theorem favorable_pairings_is_one :
  favorable_pairings = 1 :=
sorry

theorem probability_triangle_from_random_chords_value :
  probability_triangle_from_random_chords = 1 / 15 :=
sorry

end probability_triangle_from_random_chords_probability_is_favorable_over_total_total_pairings_calculation_favorable_pairings_is_one_probability_triangle_from_random_chords_value_l1495_149550


namespace min_value_x_plus_four_over_x_l1495_149506

theorem min_value_x_plus_four_over_x (x : ℝ) (h : x > 0) :
  x + 4 / x ≥ 4 ∧ (x + 4 / x = 4 ↔ x = 2) := by
  sorry

end min_value_x_plus_four_over_x_l1495_149506


namespace profit_increase_l1495_149599

theorem profit_increase (initial_profit : ℝ) (h : initial_profit > 0) :
  let april_profit := initial_profit * 1.2
  let may_profit := april_profit * 0.8
  let june_profit := initial_profit * 1.4399999999999999
  (june_profit / may_profit - 1) * 100 = 50 := by
sorry

end profit_increase_l1495_149599


namespace harolds_marbles_l1495_149570

theorem harolds_marbles (kept : ℕ) (friends : ℕ) (each_friend : ℕ) (initial : ℕ) : 
  kept = 20 → 
  friends = 5 → 
  each_friend = 16 → 
  initial = kept + friends * each_friend → 
  initial = 100 := by
sorry

end harolds_marbles_l1495_149570


namespace discriminant_of_specific_quadratic_l1495_149552

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation x^2 - 5x + 2 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 2 = 0

theorem discriminant_of_specific_quadratic :
  discriminant 1 (-5) 2 = 17 := by
  sorry

end discriminant_of_specific_quadratic_l1495_149552


namespace value_std_dev_below_mean_l1495_149547

def mean : ℝ := 16.2
def std_dev : ℝ := 2.3
def value : ℝ := 11.6

theorem value_std_dev_below_mean : 
  (mean - value) / std_dev = 2 := by sorry

end value_std_dev_below_mean_l1495_149547


namespace equal_triplet_solution_l1495_149596

theorem equal_triplet_solution {a b c : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (h1 : a * (a + b) = b * (b + c)) (h2 : b * (b + c) = c * (c + a)) :
  a = b ∧ b = c := by
sorry

end equal_triplet_solution_l1495_149596


namespace money_left_l1495_149519

/-- The amount of money Norris saved in September -/
def september_savings : ℕ := 29

/-- The amount of money Norris saved in October -/
def october_savings : ℕ := 25

/-- The amount of money Norris saved in November -/
def november_savings : ℕ := 31

/-- The amount of money Hugo spent on an online game -/
def online_game_cost : ℕ := 75

/-- The theorem stating how much money Norris has left -/
theorem money_left : 
  september_savings + october_savings + november_savings - online_game_cost = 10 := by
  sorry

end money_left_l1495_149519


namespace star_is_addition_l1495_149529

/-- A binary operation on real numbers satisfying (a ★ b) ★ c = a + b + c -/
def star_op (star : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, star (star a b) c = a + b + c

theorem star_is_addition (star : ℝ → ℝ → ℝ) (h : star_op star) :
  ∀ a b : ℝ, star a b = a + b :=
sorry

end star_is_addition_l1495_149529


namespace arithmetic_sequence_sum_l1495_149541

/-- Given an arithmetic sequence {a_n} with common ratio q ≠ 1,
    if a_1 * a_2 * a_3 = -1/8 and (a_2, a_4, a_3) forms an arithmetic sequence,
    then the sum of the first 4 terms of {a_n} is equal to 5/8. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n, a (n + 1) = a n * q) →
  a 1 * a 2 * a 3 = -1/8 →
  2 * a 4 = a 2 + a 3 →
  (a 1 + a 2 + a 3 + a 4 = 5/8) :=
by sorry

end arithmetic_sequence_sum_l1495_149541


namespace sum_remainder_eleven_l1495_149555

theorem sum_remainder_eleven (n : ℤ) : ((11 - n) + (n + 5)) % 11 = 5 := by
  sorry

end sum_remainder_eleven_l1495_149555


namespace function_properties_l1495_149581

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_negative (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : periodic_negative f)
  (h3 : increasing_on f (-1) 0) :
  (f 2 = f 0) ∧ (symmetric_about f 1) := by
  sorry

end function_properties_l1495_149581


namespace blue_gumdrops_after_replacement_l1495_149539

theorem blue_gumdrops_after_replacement (total : ℕ) (blue_percent : ℚ) (brown_percent : ℚ) 
  (red_percent : ℚ) (yellow_percent : ℚ) (h_total : total = 150)
  (h_blue : blue_percent = 1/4) (h_brown : brown_percent = 1/4)
  (h_red : red_percent = 1/5) (h_yellow : yellow_percent = 1/10)
  (h_sum : blue_percent + brown_percent + red_percent + yellow_percent < 1) :
  let initial_blue := ⌈total * blue_percent⌉
  let initial_red := ⌊total * red_percent⌋
  let replaced_red := ⌊initial_red * (3/4)⌋
  initial_blue + replaced_red = 60 := by
  sorry

end blue_gumdrops_after_replacement_l1495_149539


namespace min_rubles_to_win_l1495_149526

/-- Represents the state of the game machine --/
structure GameState :=
  (points : ℕ)
  (rubles : ℕ)

/-- Defines the possible moves in the game --/
inductive Move
  | AddOne
  | Double

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.AddOne => { points := state.points + 1, rubles := state.rubles + 1 }
  | Move.Double => { points := state.points * 2, rubles := state.rubles + 2 }

/-- Checks if the given state is valid (not exceeding 50 points) --/
def isValidState (state : GameState) : Prop :=
  state.points ≤ 50

/-- Checks if the given state is a winning state (exactly 50 points) --/
def isWinningState (state : GameState) : Prop :=
  state.points = 50

/-- Theorem stating that 11 rubles is the minimum amount needed to win --/
theorem min_rubles_to_win :
  ∃ (moves : List Move),
    let finalState := moves.foldl applyMove { points := 0, rubles := 0 }
    isValidState finalState ∧
    isWinningState finalState ∧
    finalState.rubles = 11 ∧
    (∀ (otherMoves : List Move),
      let otherFinalState := otherMoves.foldl applyMove { points := 0, rubles := 0 }
      isValidState otherFinalState → isWinningState otherFinalState →
      otherFinalState.rubles ≥ 11) :=
by sorry


end min_rubles_to_win_l1495_149526


namespace student_sums_correct_l1495_149569

theorem student_sums_correct (total_sums : ℕ) (wrong_ratio : ℕ) 
  (h1 : total_sums = 48) 
  (h2 : wrong_ratio = 2) : 
  ∃ (correct_sums : ℕ), 
    correct_sums + wrong_ratio * correct_sums = total_sums ∧ 
    correct_sums = 16 := by
  sorry

end student_sums_correct_l1495_149569


namespace students_taking_both_courses_l1495_149533

theorem students_taking_both_courses 
  (total : ℕ) 
  (french : ℕ) 
  (german : ℕ) 
  (neither : ℕ) 
  (h1 : total = 69) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : neither = 15) :
  french + german - (total - neither) = 9 := by
  sorry

end students_taking_both_courses_l1495_149533


namespace tree_spacing_l1495_149557

/-- Given 8 equally spaced trees along a straight road, where the distance between
    the first and fifth tree is 100 feet, the distance between the first and last
    tree is 175 feet. -/
theorem tree_spacing (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  (n - 1) * d / 4 = 175 :=
by sorry

end tree_spacing_l1495_149557


namespace fries_sold_total_l1495_149537

/-- Represents the number of fries sold -/
structure FriesSold where
  small : ℕ
  large : ℕ

/-- Calculates the total number of fries sold -/
def total_fries (f : FriesSold) : ℕ := f.small + f.large

/-- Theorem: If 4 small fries were sold and the ratio of large to small fries is 5:1, 
    then the total number of fries sold is 24 -/
theorem fries_sold_total (f : FriesSold) 
    (h1 : f.small = 4) 
    (h2 : f.large = 5 * f.small) : 
  total_fries f = 24 := by
  sorry


end fries_sold_total_l1495_149537


namespace union_equality_implies_m_value_l1495_149578

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equality_implies_m_value (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 := by
  sorry

end union_equality_implies_m_value_l1495_149578


namespace root_in_interval_l1495_149545

def f (x : ℝ) := x^3 + x - 4

theorem root_in_interval :
  ∃ (r : ℝ), r ∈ Set.Ioo 1 2 ∧ f r = 0 := by
  sorry

end root_in_interval_l1495_149545


namespace least_clock_equivalent_after_six_twelve_is_clock_equivalent_twelve_is_least_clock_equivalent_after_six_l1495_149590

def clock_equivalent (h : ℕ) : Prop :=
  (h ^ 2 - h) % 24 = 0

theorem least_clock_equivalent_after_six :
  ∀ h : ℕ, h > 6 → clock_equivalent h → h ≥ 12 :=
by sorry

theorem twelve_is_clock_equivalent : clock_equivalent 12 :=
by sorry

theorem twelve_is_least_clock_equivalent_after_six :
  ∀ h : ℕ, h > 6 → clock_equivalent h → h = 12 ∨ h > 12 :=
by sorry

end least_clock_equivalent_after_six_twelve_is_clock_equivalent_twelve_is_least_clock_equivalent_after_six_l1495_149590


namespace solution_set_f_range_of_m_l1495_149511

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Statement 1: Solution set of f(x) + x^2 - 1 > 0
theorem solution_set_f (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 := by sorry

-- Statement 2: Range of m when solution set of f(x) < g(x) is non-empty
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < g x m) → m > 4 := by sorry

end solution_set_f_range_of_m_l1495_149511


namespace visual_range_increase_l1495_149598

theorem visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 100)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 50 := by
  sorry

end visual_range_increase_l1495_149598


namespace total_carriages_l1495_149505

theorem total_carriages (euston norfolk norwich flying_scotsman : ℕ) : 
  euston = norfolk + 20 →
  norwich = 100 →
  flying_scotsman = norwich + 20 →
  euston = 130 →
  euston + norfolk + norwich + flying_scotsman = 460 := by
  sorry

end total_carriages_l1495_149505


namespace alpha_range_l1495_149585

noncomputable def f (α : Real) (x : Real) : Real := Real.log x + Real.tan α

theorem alpha_range (α : Real) (x₀ : Real) :
  α ∈ Set.Ioo 0 (Real.pi / 2) →
  x₀ < 1 →
  x₀ > 0 →
  (fun x => 1 / x) x₀ = f α x₀ →
  α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) :=
by sorry

end alpha_range_l1495_149585


namespace students_in_both_clubs_l1495_149522

theorem students_in_both_clubs
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (drama_or_science : ℕ)
  (h1 : total_students = 300)
  (h2 : drama_club = 100)
  (h3 : science_club = 140)
  (h4 : drama_or_science = 210) :
  drama_club + science_club - drama_or_science = 30 := by
  sorry

end students_in_both_clubs_l1495_149522


namespace foreign_stamps_count_l1495_149548

/-- Represents a collection of stamps -/
structure StampCollection where
  total : ℕ
  old : ℕ
  foreign_and_old : ℕ
  neither : ℕ

/-- Calculates the number of foreign stamps in a collection -/
def foreign_stamps (c : StampCollection) : ℕ :=
  c.total - c.old - c.neither + c.foreign_and_old

/-- Theorem stating the number of foreign stamps in the given collection -/
theorem foreign_stamps_count (c : StampCollection) 
  (h1 : c.total = 200)
  (h2 : c.old = 50)
  (h3 : c.foreign_and_old = 20)
  (h4 : c.neither = 80) :
  foreign_stamps c = 90 := by
  sorry

#eval foreign_stamps { total := 200, old := 50, foreign_and_old := 20, neither := 80 }

end foreign_stamps_count_l1495_149548


namespace min_sum_of_quadratic_roots_l1495_149510

theorem min_sum_of_quadratic_roots (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + 2*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 2*b*x + a = 0) :
  a + b ≥ 6 := by
sorry

end min_sum_of_quadratic_roots_l1495_149510


namespace ninth_group_number_l1495_149572

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalWorkers : ℕ
  sampleSize : ℕ
  samplingInterval : ℕ
  fifthGroupNumber : ℕ

/-- Calculates the number drawn from a specific group given the sampling parameters -/
def groupNumber (s : SystematicSampling) (groupIndex : ℕ) : ℕ :=
  s.fifthGroupNumber + (groupIndex - 5) * s.samplingInterval

/-- Theorem stating that for the given systematic sampling scenario, 
    the number drawn from the 9th group is 43 -/
theorem ninth_group_number (s : SystematicSampling) 
  (h1 : s.totalWorkers = 100)
  (h2 : s.sampleSize = 20)
  (h3 : s.samplingInterval = 5)
  (h4 : s.fifthGroupNumber = 23) :
  groupNumber s 9 = 43 := by
  sorry

end ninth_group_number_l1495_149572


namespace polynomial_coefficient_sum_l1495_149580

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  |a₀| + |a₁| + |a₃| = 41 := by
  sorry

end polynomial_coefficient_sum_l1495_149580


namespace total_length_climbed_result_l1495_149540

/-- The total length of ladders climbed by two workers in inches -/
def total_length_climbed (keaton_ladder_height : ℕ) (keaton_climbs : ℕ) 
  (reece_ladder_diff : ℕ) (reece_climbs : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - reece_ladder_diff
  let keaton_total := keaton_ladder_height * keaton_climbs
  let reece_total := reece_ladder_height * reece_climbs
  (keaton_total + reece_total) * 12

/-- Theorem stating the total length climbed by both workers -/
theorem total_length_climbed_result : 
  total_length_climbed 30 20 4 15 = 11880 := by
  sorry

end total_length_climbed_result_l1495_149540


namespace pet_store_choices_l1495_149567

def num_puppies : ℕ := 10
def num_kittens : ℕ := 8
def num_bunnies : ℕ := 12

def alice_choices : ℕ := num_kittens + num_bunnies
def bob_choices (alice_choice : ℕ) : ℕ :=
  if alice_choice ≤ num_kittens then num_puppies + num_bunnies
  else num_puppies + (num_bunnies - 1)
def charlie_choices (alice_choice bob_choice : ℕ) : ℕ :=
  num_puppies + num_kittens + num_bunnies - alice_choice - bob_choice

def total_choices : ℕ :=
  (alice_choices * bob_choices num_kittens * charlie_choices num_kittens num_puppies) +
  (alice_choices * bob_choices num_kittens * charlie_choices num_kittens num_bunnies) +
  (alice_choices * bob_choices num_bunnies * charlie_choices num_bunnies num_puppies) +
  (alice_choices * bob_choices num_bunnies * charlie_choices num_bunnies (num_bunnies - 1))

theorem pet_store_choices : total_choices = 4120 := by
  sorry

end pet_store_choices_l1495_149567


namespace hexagon_side_length_l1495_149521

/-- A regular hexagon with perimeter 42 cm has sides of length 7 cm each. -/
theorem hexagon_side_length (perimeter : ℝ) (num_sides : ℕ) (h1 : perimeter = 42) (h2 : num_sides = 6) :
  perimeter / num_sides = 7 := by
  sorry

end hexagon_side_length_l1495_149521
